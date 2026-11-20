# OpenVINO Dynamo IR experiment

This experiment adds an opt-in OpenVINO exporter using `torch.export` while retaining the default TorchScript path.
It compares five YOLO26 nano checkpoints: detection, segmentation, classification, pose, and OBB. Each is exported
with static and dynamic shapes, `batch=2`, `imgsz=640`, `quantize=32`, `nms=None`, and CPU export preparation.
There are ten IR pairs. No inference, dataset accuracy evaluation, or performance tests are performed.

## Runtime

Use an environment with this branch installed, the current PyTorch release (tested with 2.15), and an OpenVINO build supporting
`ov.convert_model(..., dynamo=True)`. The initial experiment uses PyTorch 2.11.0+cpu, Python 3.12.12, and
OpenVINO source revision `eb78adba5f9fa4c8b38792ad0028e7c32fdae1b2`
([openvinotoolkit/openvino#40055](https://github.com/openvinotoolkit/openvino/pull/40055)).
Use matching native libraries and Python bindings from that build; an installed release wheel is not a substitute
for pending prerequisite fixes.

Run from the checkout root, with an output directory outside the checkout:

```bash
python examples/python/OpenVINO-Dynamo/compare_ir.py self-test --output /absolute/path/to/comparator-check
python examples/python/OpenVINO-Dynamo/compare_ir.py matrix --output /absolute/path/to/experiment-output
```

The matrix downloads official checkpoints into the output directory's `weights` subdirectory. Each export runs in
a fresh process on a separate checkpoint copy. Logs and JSON receipts record command exit codes, checkpoint hashes,
OpenVINO and PyTorch versions, and exported artifact paths and hashes. Failed exports are reported explicitly and
do not prevent the remaining cases from running. Exit status zero from the matrix runner means collection finished;
inspect every scenario's `status` in `results.json` for export failures or differences.

To reproduce a single export:

```bash
python examples/python/OpenVINO-Dynamo/compare_ir.py export --output /absolute/path/to/experiment-output \
    --model yolo26n.pt --dynamo --dynamic
```

Omit `--dynamo` for TorchScript and `--dynamic` for static shapes.

## Equality criterion

The runner reloads the serialized IRs. Its output-rooted graph description includes operator types and versions,
attributes, port-indexed connectivity, shared-node identity, parameter/result order, element types, PartialShapes,
and SHA256 hashes of exact constant bytes. It compares naming separately. It does not rewrite, optimize, or
normalize the graphs to force a match. Unsupported nested control flow causes an explicit comparison error rather
than a claim of equality.

Each successful pair produces graph snapshots, a structural diff, and `comparison.json`. Op counts are diagnostic
only, not the equality test. A graph mismatch is not proof of numerical inequality. Matching graphs and constants
establish only the tested artifact identity, not performance or support for untested settings.

Public tensor names and deployment metadata must also be considered when interpreting the experiment; graph
equality alone is not a claim that the serialized XML/BIN or the complete deployment contract is identical.

Dynamic Dynamo export calls `torch.export.export` directly and passes the `ExportedProgram` to OpenVINO. Height and
width are declared as `stride * Dim(min=2)`, which matches the divisibility guards of concat-aligned models such as
YOLOv9 (independent `Dim.AUTO` dimensions fail in the constraint solver). Stride-1 classification models use
`Dim.AUTO(min=2)`. The resulting IR inputs are `[?,C,64..,64..]` (`128..` for P6 models); OpenVINO shapes cannot
express divisibility, so inference inputs must still be multiples of the stride, as with TorchScript.

With `nms=True`, the dynamic batch range is **1 through the configured `batch`** because embedded NMS unrolls
one branch per padded image. Larger batches are rejected by the exported model rather than silently truncated.
The Dynamo NMS path uses symbolic dimensions instead of `torch._shape_as_tensor`; TorchScript is unchanged.
RT-DETR uses symbolic TopK sizes during Dynamo capture as well. Dynamic end-to-end heads and RT-DETR require
OpenVINO symbolic `min` support; dynamic NMS also uses symbolic `max`. These paths target the current PyTorch
release (2.15), with no older-version compatibility workarounds.

On PyTorch 2.15.0+cpu and TorchVision 0.29.2+cpu, the updated exporter and OpenVINO symbolic min/max lowering
export YOLOv8n embedded NMS, YOLO26n end-to-end, and RT-DETR-l with dynamic shapes. CPU FP32 outputs were
compared to the export-prepared PyTorch models at `(2,3,640,640)`, `(1,3,320,480)`, `(1,3,64,96)`, and
`(2,3,480,320)`, using the same resized bus image, with `rtol=atol=1e-4` and exact output-shape checks.
YOLO26n correctly reduces its output to 126 detections at 64x96 instead of forcing a fixed TopK of 300.
NMS batch 3 is rejected when exported with `batch=2`. These checks do not establish dataset accuracy or latency.
The existing TorchScript end-to-end YOLO path specializes K=300 and fails at 64x96 (only 126 anchors);
the Dynamo path handles that case. This experiment does not change the default TorchScript behavior.

The focused embedded-NMS regression requires the experimental OpenVINO source build and is opt-in:

```bash
python -m pytest tests/test_exports.py --slow -k openvino_dynamo_nms_batch -q
```

It checks nonempty detections, exact output batch sizes for batches 1 and 2, and rejection of batch 3.

## Initial results

All ten TorchScript exports succeeded. All five static Dynamo exports succeeded, but none produced an identical
graph. All five dynamic Dynamo exports failed during PyTorch capture with shape-constraint violations, before
OpenVINO operation translation. No failed export was retried with static shapes or TorchScript.

| Model | Static TS/Dynamo nodes | Static comparison | Dynamic Dynamo |
| --- | --- | --- | --- |
| `yolo26n.pt` | 637 / 637 | Different graph; same shapes/types | Constraint violation |
| `yolo26n-seg.pt` | 729 / 729 | Different graph; same shapes/types | Constraint violation |
| `yolo26n-cls.pt` | 246 / 247 | Different graph; same shapes/types | Constraint violation |
| `yolo26n-pose.pt` | 706 / 706 | Different graph; same shapes/types | Constraint violation |
| `yolo26n-obb.pt` | 686 / 699 | Different graph; same shapes/types | Constraint violation |

The floating-point constant multisets match exactly in all five static pairs, including occurrence counts.
Integer shape/index constants and graph structure differ:

- TorchScript reshapes use zero-copy dimensions, e.g. `[0, 4, -1]` with `special_zero=True`. Dynamo embeds the static
  batch, e.g. `[2, 4, -1]` with `special_zero=False`.
- Detection, segmentation, and pose use slices in place of one TorchScript split and eliminate several shape
  calculation nodes. Classification replaces a `ShapeOf` with static shape constants.
- OBB replaces two splits with four slices and two gathers, with nine additional integer constants.
- Public tensor names differ (for example `x.1` versus `x`). Deployment `model_info` agrees; export metadata differs
  by timestamp and the intentional `dynamo` argument/conversion provenance.

For detection-family dynamic exports, PyTorch suggests derived dimensions such as `height = 32 * h` and
`width = 32 * w`. Classification also rejects ranges allowing spatial dimensions small enough to specialize
intermediate dimensions to one. The independent named dimensions used by the prerequisite OpenVINO conversion
path do not express those complete constraints. These failures are recorded as capture/shape-contract limitations,
not unsupported-operator failures or newly established translator bugs.

**Initial conclusion:** static export worked on the tested runtime, but IR identity was not established. No
performance tests were run.

## Follow-up results

The follow-up used #40055 Python and OVC revision `40a5ab5887693f509b4ce9f6f868364ad48195bb` with the same native
build. Its newer native changes are limited to random operators that these models do not use.

- Static FP32 IR differences are confined to shape computations. CPU execution graphs match for detection,
  segmentation, classification, and pose; OBB has one extra one-element constant. On a two-image batch, outputs are
  bit-identical for all five models.
- With bounded spatial dimensions, all five dynamic Dynamo exports succeed. Detection-family models export as
  `[?,3,64..,64..]`; classification exports as `[?,3,2..,2..]`. CPU FP32 outputs are bit-identical to TorchScript
  dynamic exports for shapes `(2,3,640,640)`, `(1,3,320,480)`, `(3,3,480,640)`, `(1,3,64,64)`, and `(1,3,1024,768)`.
- INT8 naming and dynamic-shape latency remain untested follow-up work.

## Model matrix results

The matrix covered 182 official checkpoints and 548 TorchScript/Dynamo pairs on PyTorch 2.15.0+cpu with OpenVINO
from openvinotoolkit/openvino#40055. Checkpoints included YOLOv3, v5, v6, v8, v9, v10, v11, v12 and v26;
YOLO-World and YOLO-WorldV2; YOLOE, including prompt-free models; RT-DETR; and FastSAM. All tasks were covered,
including P6 and grayscale models. The scenarios were static, dynamic, FP16 static and dynamic, NMS static and
dynamic, and end-to-end static and dynamic. SAM, SAM2, MobileSAM, YOLO-NAS and INT8 were excluded.

- No TorchScript or Dynamo export failed.
- **465** pairs are identical: 12 raw IRs, 228 after MOC and constant folding, and 225 dynamic pairs when specialized
  to `(2,C,640,640)`, `(1,C,320,448)` and `(1,C,128,192)`.
- **79** further pairs produced bit-identical CPU FP32 outputs:
  - **74 NMS pairs** differ only in integer shape arithmetic. Dynamo uses `Minimum(n, max_det)` with a scalar
    `Gather`; dynamic NMS also uses symbolic `Maximum`, `Multiply` and `Add` nodes. FastSAM also uses `Ceiling`
    to lower `trunc`.
  - **4 YOLOE-v8-seg dynamic pairs** differ because MOC folds the `BNContrastiveHead` BatchNorm and convolution
    biases in a different order. The difference is at most `1e-6` absolute in three constants. Substituting the
    TorchScript values makes the graphs identical.
  - **1 YOLOE-26 prompt-free end-to-end dynamic pair** uses `TopK k=min(N,300)`. The candidate count is
    data-dependent.
- **4 FP16 pairs** are RT-DETR static and dynamic, and World and WorldV2 static. `torch.export` folds
  input-independent computations, such as the AIFI positional embedding and World text projections, into
  constants. FP16 compression then stores these results in FP16. TorchScript instead computes them in the graph
  from FP16 seed constants. Their FP32 variants are identical, and mean and maximum errors against FP32 are
  equivalent.
- Dynamic Dynamo IRs have explicit bounds: `[?,C,64..,64..]`, with batch `..batch` for embedded NMS. TorchScript
  IRs are `[?,C,?,?]`.

Two fork changes were needed:

- `fast_nms` checks `exit_early` before `boxes.numel()`, which avoided a data-dependent guard in OBB NMS.
- Dynamic export now uses stride-derived dimensions.
