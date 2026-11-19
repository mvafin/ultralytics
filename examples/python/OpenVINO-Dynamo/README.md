# OpenVINO Dynamo IR experiment

This experiment adds an opt-in OpenVINO exporter using `torch.export` while retaining the default TorchScript path.
It compares five YOLO26 nano checkpoints: detection, segmentation, classification, pose, and OBB. Each is exported
with static and dynamic shapes, `batch=2`, `imgsz=640`, `quantize=32`, `nms=None`, and CPU export preparation.
There are ten IR pairs. No inference, dataset accuracy evaluation, or performance tests are performed.

## Runtime

Use an environment with this branch installed, PyTorch >=2.6, and an OpenVINO build supporting
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

Dynamic Dynamo export requests a dynamic batch and height/width of at least twice the model's largest stride.
`torch.export` may infer a nonzero minimum spatial size, and OpenVINO rejects a requested unrestricted dimension
that is wider than the exported range. Inference inputs must still satisfy YOLO's stride requirements.

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
