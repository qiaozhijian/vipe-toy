# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional
from vipe.utils.weights import weights_path

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


# Repo and module paths per MoGe version.
_MOGE_VARIANTS = {
    1: {"module": "moge.model.v1", "local_dir": "moge-vitl"},
    2: {"module": "moge.model.v2", "local_dir": "moge-2-vitl"},
}


def focal_length_to_fov_degrees(focal_length: float, image_width: float) -> float:
    """Compute horizontal field of view from focal length."""
    fov_rad = 2 * torch.atan(torch.tensor(image_width / (2 * focal_length)))
    fov_deg = torch.rad2deg(fov_rad)
    return fov_deg.item()


class MogeModel(DepthEstimationModel):
    """
    MoGe depth estimator. https://github.com/microsoft/MoGe

    Args:
        version: 1 for MoGe-1 (`moge.model.v1`, affine-invariant), 2 for MoGe-2
                 (`moge.model.v2`, metric scale, sharper details, faster).
                 Default: 2.
    """

    def __init__(self, version: int = 2) -> None:
        super().__init__()
        if version not in _MOGE_VARIANTS:
            raise ValueError(
                f"Unsupported MoGe version: {version}. Choose from {sorted(_MOGE_VARIANTS)}."
            )
        self.version = version
        variant = _MOGE_VARIANTS[version]

        try:
            module = __import__(variant["module"], fromlist=["MoGeModel"])
            MoGeCls = module.MoGeModel
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"moge (`{variant['module']}`) is not found in the environment. "
                "Install with: pip install git+https://github.com/microsoft/MoGe.git"
            ) from e

        # moge's from_pretrained accepts a local checkpoint file (checks Path.exists()).
        ckpt = weights_path("moge", variant["local_dir"], "model.pt")
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"MoGe checkpoint missing: {ckpt}. "
                f"Run `python scripts/eval_vipe/tools/prefetch_vipe_models.py` to download."
            )
        self.model = MoGeCls.from_pretrained(str(ckpt)).cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        # MoGe-2 is truly metric; MoGe-1 is affine-invariant but VIPE consumes it
        # as metric via fov-conditioned scale. We keep the same type for both so
        # downstream code is unchanged.
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "MoGe only supports pinhole cameras"

        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        w = rgb.shape[2]
        input_image_for_depth = rgb.moveaxis(-1, 1)

        moge_input_dict = {"fov_x": focal_length_to_fov_degrees(focal_length, w)}

        with torch.no_grad():
            moge_output_full = self.model.infer(input_image_for_depth, **moge_input_dict)

        moge_depth_hw_full = moge_output_full["depth"]
        moge_mask_hw_full = moge_output_full["mask"]

        moge_depth_tensor = torch.nan_to_num(moge_depth_hw_full, nan=1e4)
        moge_depth_tensor = torch.clamp(moge_depth_tensor, min=0, max=1e4)
        moge_depth_tensor = moge_depth_tensor * moge_mask_hw_full.float()

        if not batch_dim:
            moge_depth_tensor = moge_depth_tensor.squeeze(0)
            moge_mask_hw_full = moge_mask_hw_full.squeeze(0)

        return DepthEstimationResult(metric_depth=moge_depth_tensor)
