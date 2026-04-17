# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download paths for Segment-and-Track-Anything style checkpoints (SAM, DeAOT)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import gdown
import torch

from vipe.utils.weights import weights_path

SamBackbone = Literal["vit_b", "vit_l", "vit_h"]

# Official Segment Anything checkpoints (ViT-B / L / H).
SAM_CHECKPOINTS: dict[SamBackbone, tuple[str, str]] = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}

DEAOT_FILENAME = "R50_DeAOTL_PRE_YTB_DAV.pth"
DEAOT_GDRIVE_URL = "https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view"


def _sam_hub_dir() -> Path:
    return weights_path("sam")


def _aot_hub_dir() -> Path:
    return weights_path("deaot")


def sam_checkpoint_path(model_type: SamBackbone) -> Path:
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"Unknown SAM backbone {model_type!r}; expected one of {list(SAM_CHECKPOINTS)}")
    fname, _ = SAM_CHECKPOINTS[model_type]
    return _sam_hub_dir() / fname


def ensure_sam_checkpoint(model_type: SamBackbone) -> Path:
    """Download SAM weights if missing; return absolute path to .pth."""
    fname, url = SAM_CHECKPOINTS[model_type]
    out = _sam_hub_dir() / fname
    if not out.exists() or out.stat().st_size < 1_000_000:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(out))
    return out


def deaot_checkpoint_path() -> Path:
    return _aot_hub_dir() / DEAOT_FILENAME


def ensure_deaot_checkpoint() -> Path:
    """Download DeAOT R50 checkpoint if missing."""
    out = deaot_checkpoint_path()
    if not out.exists() or out.stat().st_size < 1_000_000:
        out.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(DEAOT_GDRIVE_URL, output=str(out), fuzzy=True)
    return out


def ensure_all_sam_sizes() -> list[Path]:
    """Prefetch every SAM variant used in configs / benchmarks."""
    return [ensure_sam_checkpoint(t) for t in SAM_CHECKPOINTS]
