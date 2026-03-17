"""
Flair3D label remapping utilities.

This module builds final semantic labels from raw PLY attributes, following the
logic used in your PointCept `flair3d` brah.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


COSIA_2_FLAIR3D = np.array(
    [0, 0, 3, 1, 1, 1, 3, 3, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32
)
LIDARHD_2_FLAIR3D = np.array([1, 1, 1, 2, 0, 3, 0, 0, 3, 3, 3], dtype=np.int32)
LIDARHD_2_COARSE_B = np.array([1, 1, 2, 2, 0, 3, 0, 0, 3, 3, 3], dtype=np.int32)

# Finer all6: building=0, greenhouse=1, impervious_surface=2, other_soil=3, herbaceous=4,
# vineyard=5, tree=6, other_infrastructure=7, void=8.
COSIA_FINER_ALL6 = np.array(
    [0, 1, 8, 2, 3, 3, 8, 8, 4, 3, 3, 5, 6, 6, 6, 8, 8, 8, 8], dtype=np.int32
)

# Finer all7: building=0, greenhouse=1, impervious_surface=2, other_soil=3, herbaceous=4,
# vineyard=5, tree=6, other_infrastructure=7 (forced from LIDARHD), agricultural_soil=8, void=9.
COSIA_FINER_ALL7 = np.array(
    [0, 1, 9, 2, 3, 3, 9, 9, 4, 8, 8, 5, 6, 6, 6, 9, 9, 9, 9], dtype=np.int32
)


lidarhd_class_dictionary = {
    1: ("#d3d3d3", "Non classé"),
    2: ("#a0522d", "Sol"),
    3: ("#b3b94d", "Végétation basse"),
    4: ("#4e9a4e", "Végétation moyenne"),
    5: ("#1f4e1f", "Végétation haute"),
    6: ("#ff0000", "Bâtiment"),
    9: ("#1e90ff", "Eau"),
    17: ("#ffff00", "Pont"),
    64: ("#ff8c00", "Sursol pérenne"),
    65: ("#8b00ff", "Artefact"),
    66: ("#000000", "Points virtuels (modélisation)"),
}

LIDARHD_NUM_CLASSES = 10
TRAINID = 0
LIDARHD_ID2TRAINID = (
    np.ones(max(lidarhd_class_dictionary.keys()) + 1, dtype=np.int32) * LIDARHD_NUM_CLASSES
)
for k, v in lidarhd_class_dictionary.items():
    if v[1] == "Non classé":
        # Keep as void.
        pass
    else:
        LIDARHD_ID2TRAINID[k] = TRAINID
        TRAINID += 1


FUSION_LABEL_REMAPS = {
    "coarse_intersection",
    "rule1",
    "inter_finerall6",
    "inter_finerall7",
}

SUPPORTED_LABEL_REMAPS = FUSION_LABEL_REMAPS


def map_labels(mapping: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return mapping[labels.astype(np.intp)]


def _require_field(attributes: Dict[str, np.ndarray], field: str) -> np.ndarray:
    if field not in attributes:
        raise KeyError(f"Required field '{field}' not found in PLY attributes")
    return attributes[field].astype(np.int32)


def _finer_mapping_from_mode(mode: str) -> np.ndarray:
    if mode == "inter_finerall6":
        return COSIA_FINER_ALL6
    if mode == "inter_finerall7":
        return COSIA_FINER_ALL7
    raise ValueError(f"Mode '{mode}' does not define a finer mapping")


def _segment_from_fusion(attributes: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    cosia = _require_field(attributes, "cosia_class")
    lidarhd = _require_field(attributes, "lidarhd_class")

    # Defensive fix: clamp unexpected lidar ids.
    lidarhd = lidarhd.copy()
    lidarhd[lidarhd > 66] = 1
    lidarhd = map_labels(LIDARHD_ID2TRAINID, lidarhd)

    coarse_cosia = map_labels(COSIA_2_FLAIR3D, cosia)
    coarse_lidarhd = map_labels(LIDARHD_2_FLAIR3D, lidarhd)

    coarse_void = 3
    agreement = coarse_cosia == coarse_lidarhd

    if mode == "coarse_intersection":
        seg = np.full(cosia.shape, coarse_void, dtype=np.int32)
        seg[agreement] = coarse_cosia[agreement]
        return seg

    if mode == "rule1":
        seg = coarse_cosia.copy()
        mask = (coarse_cosia == 1) & (coarse_lidarhd != 1) & (coarse_lidarhd != coarse_void)
        seg[mask] = coarse_lidarhd[mask]
        return seg

    finer_map = _finer_mapping_from_mode(mode)
    finer_void = int(finer_map.max())

    if mode in ("inter_finerall6", "inter_finerall7"):
        coarse_lidarhd_b = map_labels(LIDARHD_2_COARSE_B, lidarhd)
        agreement = coarse_cosia == coarse_lidarhd_b
        agreement = agreement | (lidarhd == 10)  # lidarhd void treated as agreement

    seg = np.full(cosia.shape, finer_void, dtype=np.int32)
    cosia_finer = map_labels(finer_map, cosia)
    seg[agreement] = cosia_finer[agreement]

    if mode == "inter_finerall6":
        # Other infrastructure override from LIDARHD (class 7).
        seg[lidarhd == 7] = 7
    elif mode == "inter_finerall7":
        seg[lidarhd == 7] = 7

    return seg


def build_segment(attributes: Dict[str, np.ndarray], label_definition: str) -> np.ndarray:
    if label_definition in FUSION_LABEL_REMAPS:
        return _segment_from_fusion(attributes, label_definition)
    supported = ", ".join(sorted(SUPPORTED_LABEL_REMAPS))
    raise ValueError(
        f"Unknown label_definition '{label_definition}'. Supported: {supported}"
    )

