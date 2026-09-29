"""Command-line interface: ``exaspim-axon <command>``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from exaspim_axon_segmentation.io import CcfOntology, ExaspimAsset, RegionRoi

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("exaspim_axon_segmentation")

AssetArg = Annotated[str, typer.Argument(help="Asset root, e.g. s3://aind-open-data/exaSPIM_...")]
RegionOpt = Annotated[list[str], typer.Option("--region", "-r", help="CCF ID or acronym; repeat")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """TRAILMAP axon segmentation for exaSPIM."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command("qc-region")
def qc_region(
    asset: AssetArg,
    region: RegionOpt,
    out: Annotated[Path, typer.Option(help="Output PNG")] = Path("qc/region_overlay.png"),
    detail_um: Annotated[float, typer.Option(help="XY voxel size of zoomed rows")] = 6.0,
) -> None:
    """Draw the sample-space CCF region outline on the fused image for visual confirmation."""
    from exaspim_axon_segmentation.qc import render_region_overlay

    ontology = CcfOntology.load()
    data = ExaspimAsset(asset)
    label_ids = ontology.resolve(region)
    roi = RegionRoi.from_annotation(data.annotation, label_ids)
    log.info("ROI %s: %.3f mm³, bbox %s to %s um", region, roi.volume_mm3, roi.lo_um, roi.hi_um)
    render_region_overlay(data, roi, ontology[region[0]], out, detail_um=detail_um)
    log.info("Wrote %s", out)
