import zlib
import numpy as np
import os
import matplotlib.pyplot as plt 
from time import time
from PIL import Image
from tqdm import tqdm

def crc32(fileName, file=False):
    if(file):
        fh = open(fileName, 'rb')
    else:
        fh = (fileName)
    hash = 0
    while True:
        s = fh.read(65536)
        if not s:
            break
        hash = zlib.crc32(s, hash)
    
    if(file):
        fh.close()
    return "%08X" % (hash & 0xFFFFFFFF)

def _index(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    denom = x + y
    # avoid divide-by-zero
    out = np.zeros_like(denom, dtype=float)
    mask = denom != 0
    out[mask] = (x[mask] - y[mask]) / denom[mask]
    return out

def _clip(a):
    return np.clip(a, 0.0, 1.0)

def cloud_detect_and_adjust(B02, B03, B04, B11, gain=2.5):
    """
    Inputs: B02, B03, B04, B11 can be scalars or numpy arrays (same shape).
    Returns: (cloud_mask, rgb) where
      - cloud_mask: boolean array (True = cloud detected)
      - rgb: float array shaped (..., 3) with adjusted RGB values
    """
    B02 = np.asarray(B02, dtype=float)
    B03 = np.asarray(B03, dtype=float)
    B04 = np.asarray(B04, dtype=float)
    B11 = np.asarray(B11, dtype=float)

    bRatio = (B03 - 0.175) / (0.39 - 0.175)
    NDGR = _index(B03, B04)

    mask1 = (B11 > 0.1) & (bRatio > 1)                 # first cloud rule
    mask2 = (B11 > 0.1) & (bRatio > 0) & (NDGR > 0) & (~mask1)  # second cloud rule
    cloud_mask = mask1 | mask2

    # default (non-cloud) RGB
    r = gain * B04
    g = gain * B03
    b = gain * B02

    # apply mask1: [0.5*clip(B04), 0.5*clip(B03), 0.5*clip(B02) + v]
    if np.any(mask1):
        v1 = 0.5 * (bRatio - 1)
        r[mask1] = 0.5 * _clip(B04[mask1])
        g[mask1] = 0.5 * _clip(B03[mask1])
        b[mask1] = 0.5 * _clip(B02[mask1]) + v1[mask1]

    # apply mask2: [0.5*clip(B04) + v, 0.5*clip(B03), 0.5*clip(B02)]
    if np.any(mask2):
        v2 = 5.0 * np.sqrt(bRatio * NDGR)
        r[mask2] = 0.5 * _clip(B04[mask2]) + v2[mask2]
        g[mask2] = 0.5 * _clip(B03[mask2])
        b[mask2] = 0.5 * _clip(B02[mask2])

    rgb = np.stack([r, g, b], axis=-1)
    return cloud_mask, rgb

def count_white_black_pixels(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    # white_mask = (img[:, :, 0] >= 250) & (img[:, :, 1] >= 250) & (img[:, :, 2] >= 250)
    # white_pixels = np.sum(white_mask)
    white_pixels = 0
    black_mask = (img[:, :, 0] <= 5) & (img[:, :, 1] <= 5) & (img[:, :, 2] <= 5)
    black_pixels = np.sum(black_mask)

    return white_pixels, black_pixels

def get_invalid_img(file_path):
    white = np.array([])
    black = np.array([])
    img_files = (os.listdir(file_path))
    thres = (21 * 21) // 2
    for img in tqdm(img_files):
        _, black_pixels = count_white_black_pixels(os.path.join(file_path,img))
        black = np.append(black, black_pixels)
        # white = np.append(white, white_pixels)

    # x = white >= thres
    y = black >= 3

    # indices_white = np.where(x)[0].tolist()
    indices_black = np.where(y)[0].tolist()
    
    # invalid = [img_files[idx] for idx in indices_black] + [img_files[idx] for idx in indices_white]
    invalid = [img_files[idx] for idx in indices_black] 
    remove_func = lambda item: item.removeprefix("s2_").removesuffix(".png")
    invalid = list(set(map(remove_func, invalid)))
    print(len(indices_black))
    # print(len(indices_white))
    
    return invalid

def find_largest_pair_difference(list1, list2):
    max_diff = 0.0
    max_idx = 0
    for i in range(len(list1)):
        diff = abs(list1[i] - list2[i])
        if diff > max_diff:
            max_diff = diff
            max_idx = i
    return float(max_diff), max_idx


def find_pair_difference_threshold(list1, list2, thresh):
    idx = []
    diff = []
    length = len(list1)
    for i in range(length):
        current_diff = abs(list1[i] - list2[i])
        if current_diff > thresh:
            diff.append(list1[i] - list2[i])
            idx.append(i)
    return [float(x) for x in diff], idx

if __name__ == "__main__":
    import pandas as pd
    # img1 = Image.open("s2_10x10_pngs/s2_487.png")
    # img1 = img1.convert('RGB')
    # h, w, _ = np.shape(img1)
    # img = np.asarray(img1)
    # white_pixels = 0
    # for i in range(h):
    #     for j in range(w):
    #         if(img[i,j,0] <= 5 and img[i,j,1] <= 5 and img[i,j,2] <= 5):
    #             white_pixels += 1
    # white_pixels = np.sum(img <= 5)
    # print(white_pixels)
    # x=0
    a = get_invalid_img("data/s2_img/10x10/true")
    # print(a)
    # print(len(a))
    new_idx = list(map(lambda x: int(x), a))
    # print(min(new_idx))
    path = ""
    image_path = "data/s2_img/10x10/true"
    csv_path = "data/outputs/gedi_valid_all_layers_from_h5.csv"

    df = pd.read_csv(path+csv_path)
    print(np.shape(df))
    df = df.drop(new_idx).reset_index(drop=True)
    print(np.shape(df))
    image_files = os.listdir(path+image_path)
    files_sorted = sorted(image_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    # print(files_sorted)
    new_idx = np.array(new_idx)
    files_sorted = np.array(files_sorted)
    x= np.delete(files_sorted, new_idx)
    x = x.tolist()
    # print(x)
    # print(len(x))
    # white = []
    # black = []
    # img_files = os.listdir("s2_10x10_pngs/")
    # for img in img_files:
    #     black.append(count_black_pixels(os.path.join("s2_10x10_pngs/",img)))
    #     # if(count > 0):
    #     #     one += 1
    #     # else:
    #     #     zero += 1

    # for img in img_files:
    #     white.append(count_white_pixels(os.path.join("s2_10x10_pngs/",img)))
    #     # if(count > 0):
    #     #     one += 1
    #     # else:
    #     #     zero += 1


    # white = np.array(white) #70
    # x = ((white >= 70) & (white <= 1000))
    # indices = np.where(x)[0].tolist()
    # print([img_files[idx] for idx in indices])
    # print([white[idx] for idx in indices])

    # black = np.array(black) #70
    # x = ((black >= 10) & (black <= 500)) #
    # indices = np.where(x)[0].tolist()
    # print([img_files[idx] for idx in indices])
    # print([black[idx] for idx in indices])
        
    # Both less than 500 is fine

    # a = time()
    # x = crc32("model/data/model/data/s2_6933/merged/merged_B03_6933.tif", file=True)
    # print(x)
    # print(time()-a)