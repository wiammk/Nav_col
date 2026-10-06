import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sanitize_run_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return name.strip("._") or "default"


@dataclass(frozen=True)
class RunLayout:
    """All generated paths belonging to one building."""

    run_dir: Path

    @property
    def processed(self) -> Path:
        return self.run_dir / "data" / "processed"

    @property
    def graph(self) -> Path:
        return self.processed / "graph.gpickle"

    @property
    def models(self) -> Path:
        return self.run_dir / "models"

    @property
    def local_models(self) -> Path:
        return self.models / "qlearning"

    @property
    def centralized_models(self) -> Path:
        return self.models / "centralized" / "qlearning"

    @property
    def results(self) -> Path:
        return self.run_dir / "results"

    @property
    def federated_results(self) -> Path:
        return self.results / "federated"

    @property
    def evaluation_results(self) -> Path:
        return self.run_dir / "evaluation" / "results"

    @property
    def evaluation_plots(self) -> Path:
        return self.run_dir / "evaluation" / "plots"

    @property
    def visualizations(self) -> Path:
        return self.run_dir / "visualizations"

    def create(self) -> "RunLayout":
        for path in (
            self.processed,
            self.models,
            self.results,
            self.evaluation_results,
            self.evaluation_plots,
            self.visualizations,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def layout_from_run_dir(run_dir: str | Path) -> RunLayout:
    return RunLayout(resolve_project_path(run_dir))


def run_dir_from_graph(graph_path: str | Path) -> Path | None:
    """Return runs/<building> when the graph already belongs to a run."""
    graph = resolve_project_path(graph_path).resolve()
    for parent in graph.parents:
        if parent.parent.name.lower() == "runs":
            return parent
    return None


def layout_for_graph(graph_path: str | Path, fallback_name: str = "default") -> RunLayout:
    run_dir = run_dir_from_graph(graph_path)
    if run_dir is None:
        run_dir = RUNS_ROOT / sanitize_run_name(fallback_name)
    return RunLayout(run_dir)


def latest_run_layout() -> RunLayout:
    """Choose the most recently updated building for standalone report scripts."""
    candidates = (
        [path for path in RUNS_ROOT.iterdir() if path.is_dir()]
        if RUNS_ROOT.exists()
        else []
    )
    if not candidates:
        return RunLayout(RUNS_ROOT / "default")
    return RunLayout(max(candidates, key=lambda path: path.stat().st_mtime))
