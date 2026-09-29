from pathlib import Path

import numpy as np
import pytest
import zarr


def write_ngff(path: Path, levels: list[tuple[np.ndarray, list[float], list[float]]]) -> str:
    """Write a minimal OME-NGFF v0.4 multiscale (t, c, z, y, x) group."""
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    datasets = []
    for i, (data, scale, translation) in enumerate(levels):
        group.create_array(str(i), data=data[None, None], chunks=(1, 1, 4, 4, 4))
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, *scale]},
                    {"type": "translation", "translation": [0.0, 0.0, *translation]},
                ],
            }
        )
    axes = [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}] + [
        {"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"
    ]
    group.attrs["multiscales"] = [{"version": "0.4", "axes": axes, "datasets": datasets}]
    return str(path)


@pytest.fixture
def labels_and_image(tmp_path: Path) -> tuple[str, str, np.ndarray]:
    """An 8 um label volume with a cube of label 7, and a 4 um image offset by half a voxel."""
    labels = np.zeros((6, 6, 6), dtype=np.uint32)
    labels[2:4, 1:5, 3:5] = 7
    coarse = labels[::2, ::2, ::2]
    labels_uri = write_ngff(
        tmp_path / "labels.zarr",
        [(labels, [8.0, 8.0, 8.0], [0.0, 0.0, 0.0]), (coarse, [16.0, 16.0, 16.0], [4, 4, 4])],
    )
    image = np.arange(12**3, dtype=np.uint16).reshape(12, 12, 12)
    image_uri = write_ngff(tmp_path / "image.zarr", [(image, [4.0, 4.0, 4.0], [-2.0, -2.0, -2.0])])
    return labels_uri, image_uri, labels
