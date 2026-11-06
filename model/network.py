import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks — unchanged from original
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: conditions spatial features on a vector."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feat_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(cond).chunk(2, dim=1)          # [B, C]
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        return self.act(out + identity)


# ---------------------------------------------------------------------------
# Encoder — unchanged from original
# ---------------------------------------------------------------------------

class S2EncoderFCN_ResFiLM(nn.Module):
    """
    Sentinel-2 patch encoder with residual blocks and FiLM conditioning
    from the AlphaEarth embedding.

    Input:
        s2     : [B, in_channels, 21, 21]
        ae_vec : [B, ae_dim]          (AlphaEarth + snow_frac + cloud_frac)

    Output:
        feat_map : [B, embed_dim, ~11, ~11]
    """

    def __init__(self, in_channels: int = 10, embed_dim: int = 128, ae_dim: int = 64):
        super().__init__()
        self.use_ae = ae_dim is not None

        # 21×21 → keep
        self.block1 = ResBlock(in_channels, 32, stride=1)
        # downsample once: 21×21 → 11×11
        self.block2 = ResBlock(32, 64, stride=2)
        # keep 11×11
        self.block3 = ResBlock(64, 128, stride=1)

        self.fc1x1 = nn.Conv2d(128, embed_dim, kernel_size=1)

        if self.use_ae:
            self.film1 = FiLM(ae_dim, 32)
            self.film2 = FiLM(ae_dim, 64)
            self.film3 = FiLM(ae_dim, embed_dim)

    def forward(
        self,
        s2: torch.Tensor,
        ae_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.block1(s2)
        if self.use_ae:
            x = self.film1(x, ae_vec)

        x = self.block2(x)
        if self.use_ae:
            x = self.film2(x, ae_vec)

        x = self.block3(x)
        x = self.fc1x1(x)               # [B, embed_dim, ~11, ~11]
        if self.use_ae:
            x = self.film3(x, ae_vec)

        return x


# ---------------------------------------------------------------------------
# Original single-head model — kept intact for backwards compatibility
# ---------------------------------------------------------------------------

class AGBDNet(nn.Module):
    """
    Original single-output AGBD predictor.
    Output is in log-space; caller applies exp() to get linear AGBD.
    """

    def __init__(self, s2_encoder: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder = s2_encoder
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        s2: torch.Tensor,
        ae_vec: torch.Tensor,
    ) -> torch.Tensor:
        feat_map = self.encoder(s2, ae_vec)
        h, w = feat_map.shape[-2:]
        center_feat = feat_map[:, :, h // 2, w // 2]       # [B, embed_dim]
        return self.head(center_feat).squeeze(-1)            # [B]


# ---------------------------------------------------------------------------
# NEW dual-head model
# ---------------------------------------------------------------------------

class AGBDNet_DualHead(nn.Module):
    """
    Dual-output AGBD predictor that separately estimates leaf and wood biomass.

    Architecture:
        shared encoder  (S2EncoderFCN_ResFiLM, unchanged)
            ↓ center feature [B, embed_dim]
        shared trunk  (Linear → ReLU)
            ├─ leaf head  : spectral-only path  → leaf AGBD  (linear space)
            └─ wood head  : trunk + AE re-injection → wood AGBD  (linear space)
        total AGBD = leaf + wood

    Motivation:
        - Sentinel-2 spectral bands are leaf-dominated → leaf head relies on
          shared trunk (which encodes spectral features).
        - AlphaEarth embeddings encode canopy structure → wood head re-injects
          AE vector directly as a structural prior.
        - Softplus on both heads enforces non-negative biomass predictions.

    Output (forward):
        dict with keys:
            'leaf'  : [B]  leaf AGBD  (linear space, Mg/ha)
            'wood'  : [B]  wood AGBD  (linear space, Mg/ha)
            'total' : [B]  leaf + wood  (linear space, Mg/ha)
    """

    def __init__(
        self,
        s2_encoder: nn.Module,
        embed_dim: int,
        ae_dim: int = 66,
    ):
        super().__init__()
        self.encoder = s2_encoder

        # Shared trunk: common spectral + structural features
        self.shared_trunk = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
        )

        # ---- Leaf head ----
        # Relies on shared trunk (spectrally rich after FiLM conditioning).
        # Softplus keeps output strictly positive.
        self.leaf_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        # ---- Wood head ----
        # Re-injects the AlphaEarth embedding to reinforce structural signal.
        # Input: concat(shared_trunk_out [128], ae_projection [128]) → 256 dims
        self.wood_ae_proj = nn.Sequential(
            nn.Linear(ae_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.wood_head = nn.Sequential(
            nn.Linear(128 + 128, 64),       # trunk + AE projection
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        s2: torch.Tensor,
        ae_vec: torch.Tensor,
    ) -> dict:
        # Encode: S2 patch + AlphaEarth FiLM conditioning
        feat_map = self.encoder(s2, ae_vec)             # [B, embed_dim, H, W]
        h, w = feat_map.shape[-2:]
        center_feat = feat_map[:, :, h // 2, w // 2]   # [B, embed_dim]

        # Shared trunk
        shared = self.shared_trunk(center_feat)         # [B, 128]

        # Leaf biomass: spectral features dominate
        leaf_biomass = self.leaf_head(shared).squeeze(-1)           # [B]

        # Wood biomass: structural AE features re-injected
        ae_proj      = self.wood_ae_proj(ae_vec)                    # [B, 128]
        wood_biomass = self.wood_head(
            torch.cat([shared, ae_proj], dim=1)
        ).squeeze(-1)                                               # [B]

        total_agbd = leaf_biomass + wood_biomass                    # [B]

        return {
            "leaf":  leaf_biomass,
            "wood":  wood_biomass,
            "total": total_agbd,
        }
