"""Run the two post-review experiments on the saved ten-seed protocols."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "runs" / "Office_Building" / "data" / "processed" / "graph.gpickle"
BASE = ROOT / "runs" / "Office_Building" / "experiments"
OUT = ROOT / "runs" / "Office_Building" / "review_additions"
SEEDS = tuple(range(42, 52))
FLEETS = (1, 3, 5, 10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log_path):
    print(f"Running: {subprocess.list2cmdline([str(x) for x in command])}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline([str(x) for x in command]) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Finished: {command[1]}", flush=True)


def evaluate(method, robots, protocol_path, output_dir, model_path=None):
    summary = output_dir / f"{json.loads(protocol_path.read_text(encoding='utf-8')).get('regime', 'static')}_summary.csv"
    if summary.exists():
        return
    command = [
        sys.executable,
        "evaluation/evaluate_review_variant.py",
        "--graph", str(GRAPH),
        "--protocol", str(protocol_path),
        "--robots", str(robots),
        "--method", method,
        "--output-dir", str(output_dir),
        "--max-steps", "200",
        "--seed", "33003",
    ]
    if model_path:
        command.extend(["--model", str(model_path)])
    run(command, output_dir / "evaluation.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not GRAPH.exists():
        raise FileNotFoundError(GRAPH)

    protocols = {}
    for robots in FLEETS:
        protocols[robots] = {}
        for name in ("fixed_test_protocol.json", "dynamic_test_protocol.json"):
            paths = [
                BASE / f"seed_{seed}" / f"robots_{robots}" / "evaluation" / name
                for seed in SEEDS
            ]
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError("Missing frozen protocol(s): " + ", ".join(missing))
            hashes = {sha256(path) for path in paths}
            if len(hashes) != 1:
                raise ValueError(f"Saved {name} differs across training seeds for K={robots}.")
            protocols[robots][name] = paths[0]

    manifest_path = OUT / "review_experiment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "post-review visit-weighted FedAvg and shortest-path baseline",
        "seeds": list(SEEDS),
        "visit_weighted_robot_counts": [3, 5, 10],
        "shortest_path_robot_counts": list(FLEETS),
        "rounds": 20,
        "episodes_per_round_per_client": 50,
        "max_steps": 200,
        "protocol_hashes": {
            str(k): {name: sha256(path) for name, path in files.items()}
            for k, files in protocols.items()
        },
        "runs": [],
    }
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runs"] = previous.get("runs", [])

    for seed in SEEDS:
        for robots in (3, 5, 10):
            run_dir = OUT / "visitation_weighted" / f"seed_{seed}" / f"robots_{robots}"
            model_path = run_dir / "global_qtable_final.pkl"
            metadata_path = run_dir / "run_metadata.json"
            if not (args.resume and model_path.exists() and metadata_path.exists()):
                command = [
                    sys.executable,
                    "scripts/training/train_federated.py",
                    "--algo", "qlearning",
                    "--q-aggregation", "visitation_weighted",
                    "--graph", str(GRAPH),
                    "--robots", str(robots),
                    "--rounds", "20",
                    "--episodes", "50",
                    "--max-steps", "200",
                    "--eval-episodes", "50",
                    "--validation-seed", "22002",
                    "--seed", str(seed),
                    "--save-every", "5",
                    "--heterogeneity", "iid",
                    "--alpha", "0.1",
                    "--gamma", "0.99",
                    "--epsilon-start", "1.0",
                    "--epsilon-min", "0.05",
                    "--epsilon-decay", "0.995",
                    "--save-dir", str(run_dir),
                ]
                run(command, run_dir / "train.log")
            for name, protocol_path in protocols[robots].items():
                evaluate(
                    "qtable",
                    robots,
                    protocol_path,
                    run_dir / "evaluation" / Path(name).stem,
                    model_path=model_path,
                )
            manifest["runs"].append({
                "variant": "visitation_weighted_fedavg_qlearning",
                "seed": seed,
                "robots": robots,
                "output_dir": str(run_dir),
            })
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for robots in FLEETS:
        for name, protocol_path in protocols[robots].items():
            evaluate(
                "shortest_path",
                robots,
                protocol_path,
                OUT / "shortest_path" / f"robots_{robots}" / Path(name).stem,
            )
        manifest["runs"].append({
            "variant": "weighted_shortest_path_policy",
            "seed": None,
            "robots": robots,
            "output_dir": str(OUT / "shortest_path" / f"robots_{robots}"),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Completed review experiments. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
