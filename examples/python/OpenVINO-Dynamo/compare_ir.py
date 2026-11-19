"""Reproduce the FP32 YOLO26 TorchScript versus Dynamo OpenVINO IR experiment."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import openvino as ov
import yaml

MODELS = ("yolo26n.pt", "yolo26n-seg.pt", "yolo26n-cls.pt", "yolo26n-pose.pt", "yolo26n-obb.pt")


def save_json(path, value):
    """Write a reproducible JSON artifact."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    """Hash a file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def port_info(port):
    """Describe the public tensor contract without its exporter-specific names."""
    return {"type": str(port.get_element_type()), "shape": str(port.get_partial_shape())}


def rt_value(value):
    """Unwrap nested OpenVINO model metadata without truncation."""
    if isinstance(value, ov.OVAny):
        return rt_value(value.value)
    if hasattr(value, "items"):
        return {key: rt_value(item) for key, item in value.items()}
    return value


def snapshot(model):
    """Describe the output-rooted DAG, preserving input ports and shared nodes."""
    queue = []
    indices = {}

    def index(node):
        key = node.get_instance_id()
        if key not in indices:
            indices[key] = len(queue)
            queue.append(node)
        return indices[key]

    roots = [index(node) for node in model.get_results()]
    parameters = [index(node) for node in model.get_parameters()]
    sinks = [index(node) for node in model.get_sinks()]
    nodes, names = [], []
    for node in queue:
        kind = node.get_type_name()
        if kind in {"If", "Loop", "TensorIterator"}:
            raise NotImplementedError(f"Nested {kind} graphs require body and port-mapping comparison.")
        info = node.get_type_info()
        record = {
            "op": kind,
            "version": info.version_id,
            "attributes": node.get_attributes(),
            "inputs": [
                [index(port.get_source_output().get_node()), port.get_source_output().get_index()]
                for port in node.inputs()
            ],
            "outputs": [port_info(port) for port in node.outputs()],
        }
        if kind == "Constant":
            data = node.get_data()
            record["constant"] = {
                "dtype": data.dtype.str,
                "shape": list(data.shape),
                "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            }
        nodes.append(record)
        names.append(
            {"friendly": node.get_friendly_name(), "tensors": [sorted(port.get_names()) for port in node.outputs()]}
        )
    return {
        "graph": {"results": roots, "parameters": parameters, "sinks": sinks, "nodes": nodes},
        "names": names,
        "contract": {"inputs": [port_info(p) for p in model.inputs], "outputs": [port_info(p) for p in model.outputs]},
        "public_names": {
            "inputs": [sorted(p.get_names()) for p in model.inputs],
            "outputs": [sorted(p.get_names()) for p in model.outputs],
        },
        "rt_info": rt_value(model.get_rt_info()),
        "op_counts": dict(sorted(Counter(n["op"] for n in nodes).items())),
    }


def compare(left, right, directory):
    """Compare graph semantics exactly, reporting names and contract independently."""
    directory.mkdir(parents=True, exist_ok=True)
    a, b = snapshot(left), snapshot(right)
    save_json(directory / "torchscript.graph.json", a)
    save_json(directory / "dynamo.graph.json", b)
    diff = difflib.unified_diff(
        json.dumps(a["graph"], indent=2, sort_keys=True).splitlines(keepends=True),
        json.dumps(b["graph"], indent=2, sort_keys=True).splitlines(keepends=True),
        fromfile="torchscript",
        tofile="dynamo",
    )
    (directory / "graph.diff").write_text("".join(diff))
    result = {
        "graph_equal": a["graph"] == b["graph"],
        "contract_equal": a["contract"] == b["contract"],
        "names_equal": a["names"] == b["names"],
        "public_names_equal": a["public_names"] == b["public_names"],
        "rt_info_equal": a["rt_info"] == b["rt_info"],
        "torchscript_ops": a["op_counts"],
        "dynamo_ops": b["op_counts"],
        "torchscript_contract": a["contract"],
        "dynamo_contract": b["contract"],
    }
    save_json(directory / "comparison.json", result)
    return result


def export(args):
    """Export one isolated checkpoint copy through the public Ultralytics API."""
    import torch

    from ultralytics import YOLO

    torch.manual_seed(0)
    torch.set_num_threads(2)
    weights = args.output / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    source = weights / args.model
    YOLO(str(source))  # Download once; subsequent exports use the identical checkpoint bytes.
    case = args.output / f"{Path(args.model).stem}-{'dynamic' if args.dynamic else 'static'}"
    destination = case / ("dynamo" if args.dynamo else "torchscript")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / args.model
    shutil.copy2(source, checkpoint)
    path = YOLO(str(checkpoint)).export(
        format="openvino",
        dynamo=args.dynamo,
        dynamic=args.dynamic,
        imgsz=640,
        batch=2,
        quantize=32,
        nms=None,
        device="cpu",
    )
    xml = next(Path(path).glob("*.xml"))
    save_json(
        destination / "export.json",
        {
            "xml": str(xml.resolve()),
            "checkpoint_sha256": digest(source),
            "xml_sha256": digest(xml),
            "bin_sha256": digest(xml.with_suffix(".bin")),
            "torch": torch.__version__,
            "openvino": ov.__version__,
            "openvino_module": ov.__file__,
            "experiment_script_sha256": digest(Path(__file__)),
            "dynamo": args.dynamo,
            "dynamic": args.dynamic,
        },
    )


def matrix(args):
    """Run all pairs in fresh processes, retaining every exit status and log."""
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for model in MODELS:
        for dynamic in (False, True):
            case = args.output / f"{Path(model).stem}-{'dynamic' if dynamic else 'static'}"
            case.mkdir(parents=True, exist_ok=True)
            result = {"model": model, "dynamic": dynamic, "exports": []}
            for dynamo in (False, True):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "export",
                    "--output",
                    str(args.output.resolve()),
                    "--model",
                    model,
                ]
                if dynamic:
                    command.append("--dynamic")
                if dynamo:
                    command.append("--dynamo")
                label = "dynamo" if dynamo else "torchscript"
                with (case / f"{label}.log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                result["exports"].append({"command": command, "exit_code": completed.returncode, "route": label})
            if all(item["exit_code"] == 0 for item in result["exports"]):
                exports = [
                    json.loads((case / label / "export.json").read_text()) for label in ("torchscript", "dynamo")
                ]
                if exports[0]["checkpoint_sha256"] != exports[1]["checkpoint_sha256"]:
                    raise RuntimeError(f"Checkpoint changed between exports: {case}")
                result.update(compare(*(ov.Core().read_model(e["xml"]) for e in exports), case))
                metadata = [yaml.safe_load(Path(e["xml"]).with_name("metadata.yaml").read_text()) for e in exports]
                save_json(case / "metadata.json", dict(zip(("torchscript", "dynamo"), metadata)))
                result["metadata_yaml_equal"] = metadata[0] == metadata[1]
                result["status"] = "match" if result["graph_equal"] and result["contract_equal"] else "different"
            else:
                result["status"] = "export_failed"
            results.append(result)
            save_json(args.output / "results.json", results)
            print(f"{model} dynamic={dynamic}: {result['status']}", flush=True)


def self_test(directory):
    """Reject changed constants, attributes, connectivity, shape contracts, and DAG sharing."""
    op = ov.opset13
    x, y = op.parameter([2, 3], np.float32), op.parameter([2, 3], np.float32)
    base = ov.Model([op.add(x, op.constant(np.float32(1)))], [x, y])
    renamed = base.clone()
    for node in renamed.get_ordered_ops():
        node.set_friendly_name(f"renamed_{node.get_instance_id()}")
    assert snapshot(base)["graph"] == snapshot(renamed)["graph"]
    for changed in (
        ov.Model([op.add(x, op.constant(np.float32(2)))], [x, y]),
        ov.Model([op.add(y, op.constant(np.float32(1)))], [x, y]),
        ov.Model([op.softmax(x, 0)], [x]),
    ):
        assert snapshot(base)["graph"] != snapshot(changed)["graph"]
    assert snapshot(ov.Model([op.softmax(x, 0)], [x]))["graph"] != snapshot(ov.Model([op.softmax(x, 1)], [x]))["graph"]
    shared = op.relu(x)
    assert (
        snapshot(ov.Model([op.add(shared, shared)], [x]))["graph"]
        != snapshot(ov.Model([op.add(op.relu(x), op.relu(x))], [x]))["graph"]
    )
    changed = base.clone()
    changed.reshape({0: [3, 3]})
    assert snapshot(base)["contract"] != snapshot(changed)["contract"]
    compare(base, renamed, directory)
    print("Graph comparator self-tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "matrix", "self-test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--dynamo", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "export":
        export(arguments)
    elif arguments.action == "matrix":
        matrix(arguments)
    else:
        self_test(arguments.output)
