# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for model checkpoint paths.

All third-party model weights live under ``<repo_root>/weights/<subdir>/...``
instead of the per-user ``~/.cache/torch/hub`` or HuggingFace cache. Override
the root with the ``VIPE_WEIGHTS_ROOT`` environment variable if needed.
"""

from __future__ import annotations

import os

from pathlib import Path


def weights_root() -> Path:
    env = os.environ.get("VIPE_WEIGHTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # This file lives at <repo>/project/vipe-toy/vipe/utils/weights.py
    # parents: [0]=utils, [1]=vipe, [2]=vipe-toy, [3]=project, [4]=<repo>
    return Path(__file__).resolve().parents[4] / "weights"


def weights_path(*parts: str) -> Path:
    return weights_root().joinpath(*parts)
