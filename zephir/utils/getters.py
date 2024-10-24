"""
To ease the pain of ensuring compatibility with new data structures or datasets,
this file collects key IO functions for data, metadata, and annotations
that may be edited by a user to fit their particular use case.
"""

import h5py
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch

import byotrack


# Switch default getters for our use case using byotrack
def get_slice(dataset: Path, t: int) -> np.ndarray:
    """Return a slice at specified index t.
    This should return a 4-D numpy array containing multi-channel volumetric data
    with the dimensions ordered as (C, Z, Y, X).
    """
    video = byotrack.Video(dataset / "video.tiff")
    frame = video[int(t)][..., 0]  # / 255  # Select first channel (and Normalize ?)

    if frame.ndim == 2:
        return frame[None, None]

    return frame[None]


def get_annotation_df(dataset: Path) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    data = pd.DataFrame()

    frames = json.loads((dataset / "_zephir_frames.json").read_text(encoding="utf-8"))

    video = byotrack.Video(dataset / "video.tiff")
    gt: np.ndarray = torch.load(dataset / "video_data.pt", weights_only=True)["mu"][frames].numpy()
    frame_idx = np.zeros(gt.shape[:2], dtype=np.int32)
    track_idx = np.zeros(gt.shape[:2], dtype=np.int32)
    for i, frame in enumerate(frames):
        frame_idx[i] = frame
        track_idx[i] = np.arange(gt.shape[1])

    if gt.shape[-1] == 3:
        data["x"] = gt[..., 2].ravel() / video.shape[3]
        data["y"] = gt[..., 1].ravel() / video.shape[2]
        data["z"] = gt[..., 0].ravel() / video.shape[1]
    else:
        data["x"] = gt[..., 1].ravel() / video.shape[2]
        data["y"] = gt[..., 0].ravel() / video.shape[1]
        data["z"] = np.zeros(gt.shape[:2], dtype=np.float32).ravel()

    data["t_idx"] = frame_idx.ravel()  # [0 for i in range(len(gt))]
    data["worldline_id"] = track_idx.ravel()
    data["provenance"] = np.full(gt.shape[:2], b"GT").ravel()

    return data


def get_metadata(dataset: Path) -> dict:
    """Load and return metadata for the dataset as a Python dictionary.
    This should contain at least the following:
    - shape_t
    - shape_c
    - shape_z
    - shape_y
    - shape_x
    """
    video = byotrack.Video(dataset / "video.tiff")
    if video.ndim == 5:  # TZYXC
        return {
            "shape_t": len(video),
            "shape_c": 1,
            "shape_z": video.shape[1],
            "shape_y": video.shape[2],
            "shape_x": video.shape[3],
        }
    else:  # TYXC
        return {
            "shape_t": len(video),
            "shape_c": 1,
            "shape_z": 1,
            "shape_y": video.shape[1],
            "shape_x": video.shape[2],
        }
