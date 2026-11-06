"""
Sample the AlphaEarth annual embedding at every GEDI shot and export a CSV to Google Drive.

Output columns: system:index, A00 … A63, id, .geo
(model/dataset.py reads columns [1:-2] as the 64-d embedding and matches rows on `id`.)

Usage:
    pip install earthengine-api pandas
    earthengine authenticate
    python export_alphaearth_points.py --csv outputs/gedi_valid_all_layers_from_h5_6_all.csv \
        --project YOUR_GCP_PROJECT --year 2024 --name alphaearth_vec_2024_all
"""

import argparse

import ee
import pandas as pd

CHUNK = 5000  # points per FeatureCollection built client-side


def to_feature_collection(df: pd.DataFrame) -> ee.FeatureCollection:
    feats = [
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"id": int(r.id)})
        for r in df.itertuples()
    ]
    return ee.FeatureCollection(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="GEDI shots CSV with lon, lat, id columns")
    ap.add_argument("--project", required=True, help="Google Cloud project with Earth Engine enabled")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--name", default="alphaearth_vec")
    ap.add_argument("--folder", default="alphaearth")
    args = ap.parse_args()

    ee.Initialize(project=args.project)

    shots = pd.read_csv(args.csv, usecols=["lon", "lat", "id"])
    print(f"{len(shots):,} shots")

    embedding = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{args.year}-01-01", f"{args.year + 1}-01-01")
        .mosaic()
    )

    samples = None
    for start in range(0, len(shots), CHUNK):
        fc = to_feature_collection(shots.iloc[start:start + CHUNK])
        part = embedding.sampleRegions(collection=fc, properties=["id"], scale=10, geometries=True)
        samples = part if samples is None else samples.merge(part)

    band_names = [f"A{i:02d}" for i in range(64)]
    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=args.name,
        folder=args.folder,
        fileNamePrefix=args.name,
        fileFormat="CSV",
        selectors=["system:index"] + band_names + ["id", ".geo"],
    )
    task.start()
    print(f"Started export task {task.id}; download {args.name}.csv from Drive/{args.folder} when it finishes")


if __name__ == "__main__":
    main()
