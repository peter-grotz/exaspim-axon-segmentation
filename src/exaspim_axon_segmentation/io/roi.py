"""Regions of interest defined by CCF labels in sample space."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from exaspim_axon_segmentation.io.box import Box
from exaspim_axon_segmentation.io.ngff import NgffImage


@dataclass(frozen=True)
class RegionRoi:
    """Physical bounding box of a set of CCF labels in a sample-space annotation.

    Attributes
    ----------
    label_ids : frozenset of int
        Annotation values belonging to the region (the region and all its descendants).
    lo_um, hi_um : numpy.ndarray
        Physical (z, y, x) bounds in micrometers, at voxel edges of the annotation.
    peak_um : numpy.ndarray
        Per axis, the physical position of the region's largest cross-section. Unlike the box
        center this always intersects the region (bilateral regions leave the center empty).
    n_voxels : int
        Region voxel count at the annotation's full resolution.
    voxel_volume_um3 : float
        Volume of one full-resolution annotation voxel.
    """

    label_ids: frozenset[int]
    lo_um: NDArray[np.float64]
    hi_um: NDArray[np.float64]
    peak_um: NDArray[np.float64]
    n_voxels: int
    voxel_volume_um3: float

    @property
    def volume_mm3(self) -> float:
        """Region volume in sample space (mm³)."""
        return self.n_voxels * self.voxel_volume_um3 * 1e-9

    @property
    def center_um(self) -> NDArray[np.float64]:
        """Physical center of the bounding box."""
        return (self.lo_um + self.hi_um) / 2

    @classmethod
    def from_annotation(
        cls, annotation: NgffImage, label_ids: Iterable[int], *, search_level: int | None = None
    ) -> RegionRoi:
        """Locate a region: coarse search on a low-resolution level, refine at level 0.

        Parameters
        ----------
        annotation : NgffImage
            Sample-space CCF annotation.
        label_ids : iterable of int
            Structure IDs to include (typically ``CcfOntology.resolve(...)``).
        search_level : int, optional
            Level used for the coarse search; defaults to the second-coarsest.

        Raises
        ------
        ValueError
            If no voxel carries any of the labels.
        """
        ids = frozenset(int(i) for i in label_ids)
        search_level = annotation.n_levels - 2 if search_level is None else search_level
        coarse = np.isin(annotation.read(search_level), list(ids))
        if not coarse.any():
            raise ValueError(f"None of labels {sorted(ids)} present in {annotation.uri}")

        # Pad by one coarse voxel: mode-downsampling can drop thin parts of the region.
        coarse_box = _mask_box(coarse).pad(1).clip(coarse.shape)
        lo, hi = annotation.transforms[search_level].physical_extent(coarse_box)
        fine_tf = annotation.transforms[0]
        fine_box = fine_tf.box_covering(lo, hi, annotation.shape(0))
        fine = np.isin(annotation.read(0, fine_box), list(ids))
        exact = _mask_box(fine)
        exact = Box.from_arrays(
            np.add(exact.start, fine_box.start), np.add(exact.stop, fine_box.start)
        )
        lo, hi = fine_tf.physical_extent(exact)
        peak_index = np.add(fine_box.start, _largest_sections(fine))
        peak = fine_tf.to_physical(peak_index)
        return cls(ids, lo, hi, peak, int(fine.sum()), float(np.prod(fine_tf.scale)))

    def box_in(self, image: NgffImage, level: int, *, pad_um: float = 0.0) -> Box:
        """Index-space box of this ROI (optionally padded) on a level of another image."""
        return image.transforms[level].box_covering(
            self.lo_um - pad_um, self.hi_um + pad_um, image.shape(level)
        )


def _largest_sections(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Per axis, the slice index with the most mask voxels."""
    return np.array(
        [np.argmax(mask.sum(axis=tuple(a for a in range(3) if a != axis))) for axis in range(3)]
    )


def _mask_box(mask: NDArray[np.bool_]) -> Box:
    idx = np.argwhere(mask)
    return Box.from_arrays(idx.min(axis=0), idx.max(axis=0) + 1)
