# exaSPIM × TRAILMAP — axon segmentation & skeletonization plan

Goal: run the pretrained TRAILMAP 3D U-Net (PyTorch port) on exaSPIM fused OME-Zarr, restricted
to CCF regions (reverse-transformed annotation in sample space), and produce probability maps,
skeletons, and per-region axon metrics. Keep a clean path to fine-tuning on our own annotations.

---

## 1. What exists (verified 2026-09-28)

### Model weights — `huggingface.co/dfriedma/trailmap`
| file | use |
|---|---|
| `AXON_model_210409.hdf5` | **axons — this is the one we want** |
| `FOS_model_210815.hdf5` | nuclei/cells — what the duplicated capsule uses |

- Full Keras model file (`keras_version 2.2.4-tf`): `model_config` + `model_weights` + optimizer state.
- 19,078,337 params; layer names/shapes match the capsule's `get_net()` one-to-one.
  15 Conv3D, 3 Conv3DTranspose, 14 BatchNorm.
- Input 64³ → output 36³ (valid decoder convs + fixed skip crops of 4 and 12 voxels).

### Paper (Friedmann et al. 2020, PNAS, PMID 32358193)
- Training data: LaVision Ultramicroscope II, AdipoClear; **4.06 µm XY × 3 µm Z** (plus a
  1.3 µm-XY thalamocortical set). 4 label classes: axon / background / artifact / edge; sparse
  labelling (3–10 XY planes per substack).
- Post-processing: binarize at 8 thresholds (0.2–0.9), skeletonize each, weight each skeleton by
  its threshold, sum.

### Duplicated capsule `0799359` (MD-axon-segmentation)
- A SmartSPIM H2B **cell-detection** worker (FOS model + DoG point calling), TF 2.13 / Keras 2.
- Notebook-grade: hard-coded `../data` paths, global state, commented-out threshold logic
  (`valley_pos = 1`), 300-line blockwise inference with ad-hoc z-step hacks, a vendored copy of
  `aind-zarr-utils`, and an environment full of code-server/Copilot extensions.
- Worth keeping: the architecture definition, the loss/label semantics in `training/`, and the idea
  of block-skip on empty tiles. Everything else gets rewritten.

### Data (example: `exaSPIM_851474_…_processed_2026-07-21_13-59-36`)
- `fusion/fused.zarr` — OME-NGFF v0.4, zarr v2, 8 levels, TCZYX, uint16.

  | level | Z×Y×X voxels | voxel (Z, Y, X µm) |
  |---|---|---|
  | 0 | 23606 × 43879 × 64666 | 1.0, 0.748, 0.748 |
  | 1 | ~11.8k × 21.9k × 32.3k | 2.0, 1.496, 1.496 |
  | 2 | ~5.9k × 11.0k × 16.2k | 4.0, 2.992, 2.992 |
  | 3 | 2950 × 5484 × 8083 | 8.0, 5.984, 5.984 |

- `ccf_alignment/ccf_anno_to_sample/ccf_anno_in_sample_space.zarr` — uint32 CCF structure IDs,
  **2951 × 5486 × 8084 at 8 × 5.984 × 5.984 µm, translation 0**.
  This matches the `fused_ccf_ch.zarr` grid, **not** `fused.zarr` level 3 (which is 2950 × 5484 ×
  8083 with a half-voxel translation). ⇒ region → image mapping must go through physical µm using
  each array's OME-NGFF scale + translation, never raw index × 2ⁿ.
- Channels in acquisition: 488, 561; `fused.zarr` holds the single axon-signal channel.

---

## 2. Key technical findings that shape the design

1. **Weights convert without TensorFlow.** Read HDF5 with `h5py`, map to a PyTorch module:
   - Conv3D kernel `(kD,kH,kW,in,out)` → `(out,in,kD,kH,kW)`; Conv3DTranspose
     `(kD,kH,kW,out,in)` → `(in,out,kD,kH,kW)` (no spatial flip in either framework).
   - BatchNorm `eps=1e-3` (Keras default; PyTorch default 1e-5 would silently skew outputs).
   - Preserve Keras op order **Conv → ReLU → BN** (unusual, but it's what was trained), concat order
     `[upsampled, skip]`, and channels-last → channels-first.
   - Parity: generate reference outputs once in a throwaway TF-CPU container, commit a small `.npz`
     fixture, and assert `max |Δ| < 1e-4` in CI. TF never becomes a runtime dependency.
2. **The network is fully convolutional for any input N ≡ 0 (mod 8), N ≥ 64: output = N − 28.**
   (Worked through the shape arithmetic: skip crops stay consistent.) A 64³ tile keeps only 17.8% of
   its voxels; a 192³ tile keeps 62% → **~3.5× less compute per output voxel**. Caveat: encoder
   convs use `same` zero-padding, so large tiles are not bit-identical to 64³ tiling — validate
   interior agreement empirically, then make tile size a config knob.
3. **Resolution/anisotropy gap is the main scientific risk.** Training grid is 4.06 XY × 3 Z µm;
   exaSPIM L2 is 2.99 XY × 4 Z (anisotropy *inverted*), L1 is 1.5 × 2. Plan an optional on-the-fly
   resample to a "model voxel size" (read L1/L2 → resample block → infer → resample probs back),
   and treat level + target spacing + intensity normalization as the first experiment sweep.
4. **ROI-driven I/O is what makes this tractable.** L1 is ~8 TVoxel; a handful of regions is
   typically <1%. Expand region IDs to all descendants via the CCF ontology, build the mask at 8 µm,
   compute a bbox + coarse occupancy grid, and only read/infer tiles whose (dilated) footprint hits
   the mask.

---

## 3. Target package

Repo `exaspim-axon-segmentation` (AIND internal naming, no `aind-` prefix), `src/` layout, `uv`, Python 3.12,
`torch ≥ 2.4`, ruff (100 cols), pytest, numpydoc, pinned deps, GitHub Actions CI.

```
src/exaspim_axon_segmentation/
  config.py              pydantic models: IO, ROI, inference, skeleton settings (YAML/CLI)
  cli.py                 subcommands: convert-weights | predict | skeletonize | quantify | run
  model/
    unet.py              TrailmapUNet3D (nn.Module), faithful port, typed, documented
    convert.py           Keras HDF5 → safetensors (+ model card / provenance hash)
    registry.py          named weights: trailmap-axon-210409, trailmap-fos-210815, finetuned…
  io/
    ngff.py              open OME-Zarr (anon S3), level pick by target µm, index↔physical transforms
    ontology.py          CCF structure tree: acronym/ID → descendant set
    roi.py               region mask → bbox & tile occupancy at any level (physical-space mapping)
    writer.py            OME-NGFF output (uint8-quantized probs, multiscale, same transforms)
  inference/
    tiling.py            overlap-tile plan (N in → N−28 out), edge reflect-pad, mask-aware skip
    predict.py           DataLoader w/ S3 prefetch workers, bf16 autocast, torch.compile, batching
  postprocess/
    skeleton.py          TRAILMAP multi-threshold weighted skeleton, blockwise with halos
    graph.py             skan → segments/branch points, lengths in µm, CCF label per segment
    export.py            neuroglancer precomputed skeletons + parquet tables
  quantify.py            per-region axon length, density (mm/mm³), voxel fraction
  train/                 (phase 6) sparse 4-class dataset, weighted BCE, loop, checkpoints
tests/                   parity fixture, tiling round-trip, transform & ROI tests, tiny e2e
capsule/                 thin Code Ocean wrapper: code/run → `exaspim-axon run --config …`
```

Principles: no global state, no hard-coded paths, every stage resumable and idempotent
(chunk-level skip if already written), logging that states what bounded the work, outputs carry
the same OME-NGFF transforms as inputs so they overlay in neuroglancer directly.

---

## 4. Phases

| # | Deliverable | Where | Exit criterion |
|---|---|---|---|
| 0 | Repo scaffold, CI, config/CLI skeleton | local | `uv run pytest` + ruff green in CI |
| 1 | PyTorch U-Net + weight converter + parity test | local CPU | AXON & FOS models match Keras to <1e-4 |
| 2 | NGFF I/O, ontology, ROI mapping | local, anon S3 | overlay PNG of region mask on L2/L3 crop is visually aligned |
| 3 | Inference on one small ROI | local / CO GPU | prob map in OME-Zarr; sweep L1 vs L2, resample on/off, normalization |
| 4 | Skeleton + graph + per-region metrics + neuroglancer link | local / CO | skeletons overlay raw; metrics table per region |
| 5 | Code Ocean capsule (GPU), AIND metadata (`processing.json` record) | CO | one region on 851474 and 730904 end-to-end, `has_results` true |
| 6 | Fine-tuning on exaSPIM annotations | CO GPU | held-out F1 beats pretrained baseline |

Phase 3 is the "see what this looks like" milestone; phases 1–2 are about a day of work each and
de-risk everything after.

---

## 5. Decisions (2026-09-28)

- `fused.zarr` is single-channel (the axon signal).
- First target region: **MD (362)**; its children (MDc/MDl/MDm) are not separately labelled, so all
  MD voxels carry 362. Sample-space volume 16.8 mm³, bbox 2.7 × 5.0 × 3.9 mm (z, y, x) on 851474.
- Regions come from `ccf_anno_in_sample_space.zarr` ("reverse anno"); overlay QC via
  `exaspim-axon qc-region`.
- Repo home: personal GitHub (`peter-grotz/exaspim-axon-segmentation`).
- Fine-tuning labels are SWC traces → phase 6 needs SWC → voxel label rasterization (axon core +
  radius, edge band, unlabeled elsewhere), matching TRAILMAP's sparse 4-class scheme.
