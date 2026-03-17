"""
Flair3D Dataset (LidarHD-like preprocessed scenes).

Expected directory structure under data_root:
  <data_root>/<split>/<scene_id>/

Each scene folder may contain:
- coord.npy      float32 [N, 3]
- color.npy      uint8   [N, 3] (optional; if missing, zeros are used)
- segment.npy    int32   [N]    (optional in test; if missing, filled with -1)
- strength.npy   float32 [N]    (optional; intensity normalized in preprocessing)
"""

import os

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class Flair3DDataset(DefaultDataset):
    """Dataset for Flair3D scenes preprocessed into Pointcept-style folders."""

    VALID_ASSETS = [
        "coord",
        "color",
        "strength",
        "segment",
    ]

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

