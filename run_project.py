import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import CONFIG, get_nested
from config.console import configure_console_encoding
from config.run_layout import RunLayout, resolve_project_path, sanitize_run_name


ROOT = Path(__file__).resolve().parent
ALL_ALGOS = ["qlearning", "ppo", "dqn"]


def run_cmd(args):
    print("\n> " + " ".join(str(a) for a in args))
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True, env=env)


def py(script: str):
    return [sys.executable, script]


def infer_run_name(args) -> str:
    if getattr(args, "run_name", None):
        return sanitize_run_name(args.run_name)
    if getattr(args, "ifc", None):
        return sanitize_run_name(Path(args.ifc).stem)
    if getattr(args, "synthetic", False):
        return f"synthetic_{args.floors}floors_{args.rooms}rooms"

    graph_path = Path(args.graph)
    parts = graph_path.parts
    if "runs" in parts:
        run_index = parts.index("runs")
        if run_index + 1 < len(parts):
            return sanitize_run_name(parts[run_index + 1])
    return sanitize_run_name(graph_path.stem)


def ensure_run_context(args) -> dict:
    """Create one isolated output tree for the selected building."""
    existing = getattr(args, "run_paths", None)
    if existing is not None:
        return existing

    run_name = infer_run_name(args)
    runs_dir = resolve_project_path(getattr(args, "runs_dir", "runs"))
    run_dir = runs_dir / run_name
    layout = RunLayout(run_dir)
    paths = {
        "run_dir": run_dir,
        "processed": layout.processed,
        "graph": layout.graph,
        "models": layout.models,
        "results": layout.results,
        "federated": layout.federated_results,
        "evaluation_results": layout.evaluation_results,
        "evaluation_plots": layout.evaluation_plots,
        "visualizations": layout.visualizations,
    }
    for path in paths.values():
        if isinstance(path, Path) and path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)

    source_graph = resolve_project_path(args.graph)
    building_is_being_prepared = bool(
        getattr(args, "ifc", None) or getattr(args, "synthetic", False) or getattr(args, "prepare_graph", False)
    )
    target_graph_exists = paths["graph"].exists()
    if (
        not building_is_being_prepared
        and not target_graph_exists
        and source_graph.exists()
        and source_graph.resolve() != paths["graph"].resolve()
    ):
        shutil.copy2(source_graph, paths["graph"])

    manifest_path = run_dir / "run_manifest.json"
    previous_manifest = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_manifest = {}
    source_ifc = (
        str(Path(args.ifc).resolve())
        if getattr(args, "ifc", None)
        else previous_manifest.get("source_ifc")
    )
    manifest = {
        "building": run_name,
        "source_ifc": source_ifc,
        "source_graph": str(source_graph.resolve()) if source_graph.exists() else previous_manifest.get("source_graph"),
        "single_floor": getattr(args, "single_floor", None),
        "excluded_floors": getattr(args, "exclude_floor", None) or [],
        "created_or_updated_at": datetime.now(timezone.utc).isoformat(),
        "graph_path": str(paths["graph"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    args.run_name = run_name
    args.run_dir = str(run_dir)
    args.run_paths = paths
    args.graph = str(paths["graph"])
    return paths


def prepare_graph(args):
    paths = ensure_run_context(args)
    if getattr(args, "ifc", None) and getattr(args, "synthetic", False):
        raise SystemExit("Choisissez soit --ifc, soit --synthetic, pas les deux.")
    if getattr(args, "single_floor", None) and getattr(args, "exclude_floor", None):
        raise SystemExit("Choisissez soit --single-floor, soit --exclude-floor, pas les deux.")

    if getattr(args, "ifc", None):
        ifc_path = Path(args.ifc)
        if not ifc_path.exists():
            raise SystemExit(f"Fichier IFC introuvable: {ifc_path}")
        parser_cmd = (
            py("data/ifc_parser.py") + [
                "--input", str(ifc_path),
                "--proximity", str(args.proximity),
                "--max-proximity-neighbors", str(args.max_proximity_neighbors),
                "--output-dir", str(paths["processed"]),
            ]
        )
        if getattr(args, "single_floor", None):
            parser_cmd += ["--single-floor", args.single_floor]
        for excluded_floor in getattr(args, "exclude_floor", None) or []:
            parser_cmd += ["--exclude-floor", excluded_floor]
        run_cmd(parser_cmd)
        run_cmd(
            py("data/graph_builder.py")
            + [
                "--nodes", str(paths["processed"] / "nodes.csv"),
                "--edges", str(paths["processed"] / "edges.csv"),
                "--output", str(paths["graph"]),
                "--visualize", "--source", "ifc",
            ]
        )
        return

    if getattr(args, "synthetic", False):
        run_cmd(
            py("data/ifc_parser.py")
            + [
                "--synthetic", "--floors", args.floors, "--rooms", args.rooms,
                "--output-dir", str(paths["processed"]),
            ]
        )
        run_cmd(
            py("data/graph_builder.py")
            + [
                "--nodes", str(paths["processed"] / "nodes.csv"),
                "--edges", str(paths["processed"] / "edges.csv"),
                "--output", str(paths["graph"]),
                "--visualize", "--source", "synthetic",
            ]
        )
        return

    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise SystemExit(
            f"Graphe introuvable: {graph_path}\n"
            "Ajoutez --ifc chemin\\batiment.ifc ou --synthetic pour le construire."
        )


def handle_prepare(args):
    ensure_run_context(args)
    prepare_graph(args)
    print(f"Graphe pret: {args.graph}")


def add_common_training_args(cmd, args):
    cmd += [
        "--graph",
        args.graph,
        "--robots",
        args.robots,
        "--max-steps",
        args.max_steps,
        "--seed",
        args.seed,
    ]
    return cmd


def train_local_qlearning(args):
    paths = ensure_run_context(args)
    cmd = py("scripts/training/train_qlearning.py")
    add_common_training_args(cmd, args)
    cmd += [
        "--episodes", args.episodes,
        "--save-dir", str(paths["models"] / "qlearning"),
        "--history", str(paths["results"] / "local" / "qlearning" / "local_history.csv"),
    ]
    run_cmd(cmd)


def train_centralized_qlearning(args):
    paths = ensure_run_context(args)
    cmd = py("scripts/training/train_centralized_qlearning.py")
    add_common_training_args(cmd, args)
    total_trajectories = getattr(args, "total_trajectories", None)
    if total_trajectories is None:
        total_trajectories = (
            int(args.robots)
            * int(args.rounds)
            * int(args.episodes_per_round)
        )
    cmd += [
        "--episodes", args.episodes,
        "--total-trajectories", str(total_trajectories),
        "--alpha", str(args.alpha),
        "--gamma", str(args.gamma),
        "--epsilon-start", str(args.epsilon_start),
        "--epsilon-min", str(args.epsilon_min),
        "--epsilon-decay", str(args.epsilon_decay),
        "--save-dir", str(paths["models"] / "centralized" / "qlearning"),
        "--model", str(paths["models"] / "centralized" / "qlearning" / "global_qtable.pkl"),
        "--history", str(paths["results"] / "centralized" / "qlearning" / "centralized_history.csv"),
    ]
    run_cmd(cmd)


def train_fedavg(args, algo: str):
    paths = ensure_run_context(args)
    cmd = py("scripts/training/train_federated.py")
    add_common_training_args(cmd, args)
    cmd += [
        "--algo",
        algo,
        "--rounds",
        args.rounds,
        "--episodes",
        args.episodes_per_round,
        "--save-every",
        args.save_every,
        "--validation-seed",
        str(args.validation_seed),
        "--save-dir",
        str(paths["federated"] / algo),
    ]
    if algo == "qlearning":
        cmd += [
            "--alpha", str(args.alpha),
            "--gamma", str(args.gamma),
            "--epsilon-start", str(args.epsilon_start),
            "--epsilon-min", str(args.epsilon_min),
            "--epsilon-decay", str(args.epsilon_decay),
        ]
    run_cmd(cmd)


def handle_train(args):
    ensure_run_context(args)
    if getattr(args, "prepare_graph", False):
        prepare_graph(args)

    algos = ALL_ALGOS if args.algo == "all" else [args.algo]

    if args.scope == "local":
        if args.algo not in ("qlearning", "all"):
            raise SystemExit("Le mode local est disponible seulement pour qlearning dans ce projet.")
        train_local_qlearning(args)

    if args.scope in ("centralized", "all"):
        if args.algo not in ("qlearning", "all"):
            raise SystemExit("Le mode centralise est disponible seulement pour qlearning dans ce projet.")
        train_centralized_qlearning(args)

    if args.scope in ("fedavg", "all"):
        for algo in algos:
            train_fedavg(args, algo)


def handle_evaluate(args):
    paths = ensure_run_context(args)
    if args.target in ("qlearning", "all"):
        cmd = py("evaluation/compare_qlearning.py") + [
            "--graph",
            args.graph,
            "--robots",
            args.robots,
            "--max-steps",
            args.max_steps,
            "--seed",
            args.seed,
            "--local-dir",
            str(paths["models"] / "qlearning"),
            "--centralized-model",
            str(paths["models"] / "centralized" / "qlearning" / "global_qtable.pkl"),
            "--fedavg-model",
            str(paths["federated"] / "qlearning" / "global_qtable_final.pkl"),
            "--local-history",
            str(paths["results"] / "local" / "qlearning" / "local_history.csv"),
            "--centralized-history",
            str(paths["results"] / "centralized" / "qlearning" / "centralized_history.csv"),
            "--fedavg-history",
            str(paths["federated"] / "qlearning" / "fl_history.csv"),
            "--out-dir",
            str(paths["evaluation_results"] / "qlearning"),
            "--plots-dir",
            str(paths["evaluation_plots"] / "qlearning"),
        ]
        run_cmd(cmd)
    if args.target in ("fedavg", "all"):
        cmd = py("evaluation/compare_federated_algorithms.py") + [
            "--federated-results-dir",
            str(paths["federated"]),
            "--out-dir",
            str(paths["evaluation_results"] / "federated_algorithms"),
            "--plots-dir",
            str(paths["evaluation_plots"] / "federated_algorithms"),
        ]
        if getattr(args, "episodes_per_round", None) is not None:
            cmd += ["--episodes-per-round", str(args.episodes_per_round)]
        run_cmd(cmd)
    if args.target in ("scalability", "all"):
        cmd = py("evaluation/evaluate_scalability.py") + [
            "--graph", args.graph,
            "--model", str(paths["federated"] / "qlearning" / "global_qtable_final.pkl"),
            "--output-dir", str(paths["evaluation_results"] / "scalability"),
            "--max-steps", args.max_steps,
            "--seed", args.seed,
            "--scenarios", str(args.scalability_scenarios),
            "--robot-counts", *[str(value) for value in args.robot_counts],
        ]
        run_cmd(cmd)


def visualize_trained(args, algo: str):
    paths = ensure_run_context(args)
    source = args.source or ("local" if algo == "qlearning" else "fedavg")
    cmd = py("scripts/visualization/visualize_trained_navigation.py") + [
        "--algo",
        algo,
        "--graph",
        args.graph,
        "--robots",
        args.robots,
        "--episodes",
        args.episodes,
        "--max-steps",
        args.max_steps,
        "--seed",
        args.seed,
        "--source",
        source,
        "--out-dir",
        str(paths["visualizations"] / "trained" / algo / source),
    ]
    if algo == "qlearning" and source == "local":
        cmd += ["--local-dir", str(paths["models"] / "qlearning")]
    elif source == "fedavg":
        model_name = "global_qtable_final.pkl" if algo == "qlearning" else "global_model_final.pt"
        cmd += ["--model", str(paths["federated"] / algo / model_name)]
    elif algo == "qlearning" and source == "centralized":
        cmd += ["--model", str(paths["models"] / "centralized" / "qlearning" / "global_qtable.pkl")]
    run_cmd(cmd)


def visualize_fedavg(args, algo: str):
    paths = ensure_run_context(args)
    model_name = "global_qtable_final.pkl" if algo == "qlearning" else "global_model_final.pt"
    cmd = py("scripts/visualization/visualize_federated_navigation.py") + [
        "--algo",
        algo,
        "--graph",
        args.graph,
        "--robots",
        args.robots,
        "--episodes",
        args.episodes,
        "--max-steps",
        args.max_steps,
        "--seed",
        args.seed,
        "--model",
        str(paths["federated"] / algo / model_name),
        "--out-dir",
        str(paths["visualizations"] / "federated" / algo),
    ]
    run_cmd(cmd)


def visualize_progress(args, algo: str):
    paths = ensure_run_context(args)
    cmd = py("scripts/visualization/visualize_fedavg_progress.py") + [
        "--algo",
        algo,
        "--graph",
        args.graph,
        "--robots",
        args.robots,
        "--max-steps",
        args.max_steps,
        "--seed",
        args.seed,
        "--checkpoint-dir",
        str(paths["federated"] / algo),
        "--out-dir",
        str(paths["visualizations"] / "fedavg_progress" / algo),
    ]
    run_cmd(cmd)


def handle_visualize(args):
    ensure_run_context(args)
    algos = ALL_ALGOS if args.algo == "all" else [args.algo]
    for algo in algos:
        if args.kind in ("trained", "all"):
            visualize_trained(args, algo)
        if args.kind in ("fedavg", "all"):
            visualize_fedavg(args, algo)
        if args.kind in ("progress", "all"):
            visualize_progress(args, algo)


def handle_full(args):
    ensure_run_context(args)
    prepare_graph(args)

    train_args = argparse.Namespace(**vars(args))
    train_args.scope = "centralized"
    train_args.algo = "qlearning"
    train_args.episodes = args.train_episodes
    train_args.total_trajectories = (
        int(args.robots) * int(args.rounds) * int(args.episodes_per_round)
    )
    handle_train(train_args)

    train_args.scope = "fedavg"
    train_args.algo = "all"
    handle_train(train_args)

    paths = ensure_run_context(args)
    run_cmd(py("evaluation/evaluate_professor_protocol.py") + [
        "--graph", args.graph,
        "--centralized-q", str(paths["models"] / "centralized" / "qlearning" / "global_qtable.pkl"),
        "--fedavg-q", str(paths["federated"] / "qlearning" / "global_qtable_final.pkl"),
        "--fedavg-dqn", str(paths["federated"] / "dqn" / "global_model_final.pt"),
        "--fedavg-ppo", str(paths["federated"] / "ppo" / "global_model_final.pt"),
        "--output-dir", str(paths["evaluation_results"] / "professor_protocol"),
        "--robots", args.robots,
        "--scenarios", str(args.test_scenarios),
        "--max-steps", args.max_steps,
        "--test-seed", str(args.test_seed),
        "--training-seed", args.seed,
    ])

    viz_args = argparse.Namespace(**vars(args))
    viz_args.kind = "all"
    viz_args.algo = "all"
    viz_args.source = "fedavg"
    viz_args.episodes = args.viz_episodes
    handle_visualize(viz_args)


def add_shared_args(parser):
    parser.add_argument("--graph", default=CONFIG["graph_path"])
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Dossier racine contenant un sous-dossier par batiment.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nom du dossier de cette execution. Par defaut: nom du fichier IFC.",
    )
    parser.add_argument("--robots", default=str(CONFIG["n_robots"]))
    parser.add_argument("--max-steps", default=str(CONFIG["max_steps"]))
    parser.add_argument("--seed", default=str(CONFIG["seed"]))


def add_graph_source_args(parser):
    parser.add_argument("--ifc", default=None, help="Chemin du fichier IFC a tester.")
    parser.add_argument("--synthetic", action="store_true", help="Generer un batiment synthetique.")
    parser.add_argument("--floors", default="3", help="Nombre d'etages si --synthetic.")
    parser.add_argument("--rooms", default="8", help="Nombre d'espaces par etage si --synthetic.")
    parser.add_argument("--proximity", type=float, default=8.0, help="Distance maximale des liens geometriques de secours.")
    parser.add_argument("--max-proximity-neighbors", type=int, default=3, help="Nombre maximal de liens de proximite choisis par espace.")
    parser.add_argument("--single-floor", default=None, metavar="ETAGE",
                        help="Limiter un IFC a un etage reel (nom/GUID, ou 'largest').")
    parser.add_argument(
        "--exclude-floor", action="append", default=None, metavar="ETAGE",
        help="Exclure explicitement un etage IFC non navigable (option repetable).",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Point d'entree unique du projet navigation RL/FL."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare",
        help="Construire uniquement le graphe du batiment, sans entrainement.",
    )
    add_graph_source_args(prepare)
    add_shared_args(prepare)
    prepare.set_defaults(func=handle_prepare)

    train = sub.add_parser("train", help="Entrainer local, centralise ou FedAvg.")
    train.add_argument("--scope", choices=["local", "centralized", "fedavg", "all"], default="fedavg")
    train.add_argument("--algo", choices=["qlearning", "ppo", "dqn", "all"], default="qlearning")
    train.add_argument("--episodes", default=str(get_nested("qlearning", "episodes", default=200)))
    train.add_argument("--rounds", default=str(get_nested("federated", "rounds", default=20)))
    train.add_argument(
        "--episodes-per-round",
        default=str(get_nested("federated", "episodes_per_round", default=50)),
    )
    train.add_argument("--save-every", default=str(get_nested("federated", "save_every", default=5)))
    train.add_argument("--total-trajectories", type=int, default=None)
    train.add_argument("--alpha", type=float, default=0.1)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--epsilon-start", type=float, default=1.0)
    train.add_argument("--epsilon-min", type=float, default=0.05)
    train.add_argument("--epsilon-decay", type=float, default=0.995)
    train.add_argument(
        "--validation-seed",
        type=int,
        default=get_nested("scientific_protocol", "validation_seed", default=22002),
    )
    train.add_argument(
        "--prepare-graph",
        action="store_true",
        help="Reconstruire nodes.csv, edges.csv et graph.gpickle avant entrainement.",
    )
    add_graph_source_args(train)
    add_shared_args(train)
    train.set_defaults(func=handle_train)

    evaluate = sub.add_parser("evaluate", help="Generer les comparaisons.")
    evaluate.add_argument(
        "--target", choices=["qlearning", "fedavg", "scalability", "all"], default="all"
    )
    evaluate.add_argument(
        "--episodes-per-round",
        default=str(get_nested("federated", "episodes_per_round", default=50)),
    )
    evaluate.add_argument(
        "--scalability-scenarios", type=int,
        default=get_nested("common_evaluation", "multi_robot_scenarios_per_size", default=200),
    )
    evaluate.add_argument(
        "--robot-counts", type=int, nargs="+", default=CONFIG.get("robot_counts", [1, 3, 5, 10])
    )
    add_shared_args(evaluate)
    evaluate.set_defaults(func=handle_evaluate)

    visualize = sub.add_parser("visualize", help="Generer les visualisations.")
    visualize.add_argument("--kind", choices=["trained", "fedavg", "progress", "all"], default="all")
    visualize.add_argument("--algo", choices=["qlearning", "ppo", "dqn", "all"], default="all")
    visualize.add_argument("--source", choices=["local", "centralized", "fedavg"], default=None)
    visualize.add_argument("--episodes", default=str(get_nested("visualization", "episodes", default=10)))
    add_shared_args(visualize)
    visualize.set_defaults(func=handle_visualize)

    full = sub.add_parser("full", help="Pipeline complet: entrainement, evaluation, visualisation.")
    full.add_argument("--train-episodes", default=str(get_nested("qlearning", "episodes", default=200)))
    full.add_argument("--rounds", default=str(get_nested("federated", "rounds", default=20)))
    full.add_argument(
        "--episodes-per-round",
        default=str(get_nested("federated", "episodes_per_round", default=50)),
    )
    full.add_argument("--save-every", default=str(get_nested("federated", "save_every", default=5)))
    full.add_argument("--alpha", type=float, default=0.1)
    full.add_argument("--gamma", type=float, default=0.99)
    full.add_argument("--epsilon-start", type=float, default=1.0)
    full.add_argument("--epsilon-min", type=float, default=0.05)
    full.add_argument("--epsilon-decay", type=float, default=0.995)
    full.add_argument(
        "--validation-seed",
        type=int,
        default=get_nested("scientific_protocol", "validation_seed", default=22002),
    )
    full.add_argument(
        "--test-scenarios",
        type=int,
        default=get_nested("common_evaluation", "single_robot_scenarios", default=200),
    )
    full.add_argument(
        "--test-seed",
        type=int,
        default=get_nested("scientific_protocol", "test_seed", default=33003),
    )
    full.add_argument("--viz-episodes", default=str(get_nested("visualization", "episodes", default=10)))
    full.add_argument(
        "--scalability-scenarios", type=int,
        default=get_nested("common_evaluation", "multi_robot_scenarios_per_size", default=200),
    )
    full.add_argument(
        "--robot-counts", type=int, nargs="+", default=CONFIG.get("robot_counts", [1, 3, 5, 10])
    )
    add_graph_source_args(full)
    add_shared_args(full)
    full.set_defaults(func=handle_full)

    return parser


def main():
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
