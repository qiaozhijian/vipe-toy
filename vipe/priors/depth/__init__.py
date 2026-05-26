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

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


def make_depth_model(model: str):
    if "-" not in model:
        model_name, model_sub = model, ""
    else:
        model_name, model_sub = model.split("-")

    if model_name == "metric3d":
        from .metric3d import Metric3DDepthModel

        return Metric3DDepthModel(version=2, model=model_sub)

    elif model_name == "unidepth":
        from .unidepth import UniDepth2Model

        return UniDepth2Model(type=model_sub)

    elif model_name == "moge":
        from .moge import MogeModel

        # "moge" (default → v2), "moge-v1", "moge-v2"
        if model_sub == "":
            version = 2
        elif model_sub in ("v1", "v2"):
            version = int(model_sub[1])
        else:
            raise ValueError(f"Unknown MoGe variant: {model}. Use 'moge', 'moge-v1', or 'moge-v2'.")
        return MogeModel(version=version)

    elif model_name == "dav3":
        from .dav3 import DepthAnything3Model
        from vipe.utils.weights import weights_path

        # vipe-toy: prefer local checkpoint if present (for offline / custom root via VIPE_WEIGHTS_ROOT)
        local_safetensors = weights_path("depth-anything-3", "DA3METRIC-LARGE", "model.safetensors")
        weights_arg = str(local_safetensors) if local_safetensors.is_file() else None
        return DepthAnything3Model(weights_path=weights_arg)

    else:
        raise ValueError(f"Unknown depth model: {model}")
