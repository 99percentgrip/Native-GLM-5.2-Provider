"""Lazy, bounded repository world model, impact prediction, and pre-mortems."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .metacognition import MetacognitiveAssessment
from .project_context import ProjectFacts, instruction_files
from .security import safe_context_text

MAX_NODES = 96
MAX_EDGES = 192
MAX_SEEDS = 24
MAX_SOURCE_BYTES = 128_000
MAX_PREMORTEM_ITEMS = 5
_SOURCE_SUFFIXES = {".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}
_TEST_PARTS = {"test", "tests", "spec", "specs", "__tests__"}
_PACKAGE_FILES = {
    "pyproject.toml",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "registry.json",
    "manifest.json",
}
_PLATFORM_WORDS = {
    "linux": "linux",
    "macos": "macOS",
    "darwin": "macOS",
    "windows": "Windows",
    "arm64": "ARM64",
    "aarch64": "ARM64",
    "x86_64": "x86-64",
    "amd64": "x86-64",
}
_IMPORT_PATTERNS = (
    re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE),
    re.compile(r"(?:from\s+|require\s*\(|import\s*\()[\"']([^\"']+)[\"']", re.MULTILINE),
    re.compile(r"^\s*use\s+(?:crate::)?([\w:]+)", re.MULTILINE),
)


def _relative(root: Path, value: str | Path) -> str | None:
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def _terms(task: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", task[:4000])
        if word.lower() not in {"the", "and", "for", "with", "from", "this", "that", "implement"}
    }


def _bounded_int(value: Any, maximum: int = 10_000) -> int:
    try:
        return max(0, min(maximum, int(value)))
    except (TypeError, ValueError):
        return 0


def _strings(value: Any, count: int = 40, length: int = 500) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:length] for item in value[:count]]


@dataclass(frozen=True)
class WorldNode:
    path: str
    kinds: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class WorldEdge:
    source: str
    target: str
    relation: str


@dataclass
class ImpactPrediction:
    objective_hash: str
    edit_generation: int
    files: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    packaging: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    observed_files: list[str] = field(default_factory=list)
    observed_checks: list[str] = field(default_factory=list)
    unexpected_files: list[str] = field(default_factory=list)
    compared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PremortemItem:
    failure: str
    detection: str


class RepositoryIntelligence:
    """Build only the repository slice needed by the current objective."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.objective_hash = ""
        self.slice_hash = ""
        self.nodes: dict[str, WorldNode] = {}
        self.edges: list[WorldEdge] = []
        self.failure_kinds: dict[str, int] = {}
        self.ownership: dict[str, tuple[str, ...]] = {}
        self.prediction: ImpactPrediction | None = None
        self.premortem: list[PremortemItem] = []
        self.builds = 0
        if not isinstance(data, dict):
            return
        self.objective_hash = str(data.get("objective_hash", ""))[:16]
        self.slice_hash = str(data.get("slice_hash", ""))[:16]
        self.builds = _bounded_int(data.get("builds", 0))
        raw_nodes = data.get("nodes", [])
        for raw in (raw_nodes if isinstance(raw_nodes, list) else [])[:MAX_NODES]:
            if not isinstance(raw, dict):
                continue
            path = str(raw.get("path", ""))[:500]
            if path:
                self.nodes[path] = WorldNode(
                    path,
                    tuple(str(v)[:30] for v in raw.get("kinds", [])[:6]),
                    str(raw.get("reason", ""))[:160],
                )
        raw_edges = data.get("edges", [])
        for raw in (raw_edges if isinstance(raw_edges, list) else [])[:MAX_EDGES]:
            if isinstance(raw, dict):
                self.edges.append(
                    WorldEdge(
                        str(raw.get("source", ""))[:500],
                        str(raw.get("target", ""))[:500],
                        str(raw.get("relation", ""))[:40],
                    )
                )
        raw_failures = data.get("failure_kinds", {})
        self.failure_kinds = (
            {str(key)[:40]: _bounded_int(value) for key, value in raw_failures.items()}
            if isinstance(raw_failures, dict)
            else {}
        )
        raw_ownership = data.get("ownership", {})
        self.ownership = (
            {
                str(path)[:500]: tuple(str(owner)[:100] for owner in owners[:12])
                for path, owners in raw_ownership.items()
                if isinstance(owners, list)
            }
            if isinstance(raw_ownership, dict)
            else {}
        )
        raw_prediction = data.get("prediction")
        if isinstance(raw_prediction, dict):
            self.prediction = ImpactPrediction(
                objective_hash=str(raw_prediction.get("objective_hash", ""))[:16],
                edit_generation=_bounded_int(raw_prediction.get("edit_generation", 0), 1_000_000),
                files=_strings(raw_prediction.get("files")),
                tests=_strings(raw_prediction.get("tests")),
                packaging=_strings(raw_prediction.get("packaging")),
                platforms=_strings(raw_prediction.get("platforms"), length=80),
                reasons=_strings(raw_prediction.get("reasons"), count=10, length=200),
                observed_files=_strings(raw_prediction.get("observed_files")),
                observed_checks=_strings(raw_prediction.get("observed_checks"), length=120),
                unexpected_files=_strings(raw_prediction.get("unexpected_files")),
                compared=bool(raw_prediction.get("compared", False)),
            )
        raw_premortem = data.get("premortem", [])
        for raw in (raw_premortem if isinstance(raw_premortem, list) else [])[:MAX_PREMORTEM_ITEMS]:
            if isinstance(raw, dict):
                self.premortem.append(
                    PremortemItem(
                        str(raw.get("failure", ""))[:300], str(raw.get("detection", ""))[:300]
                    )
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_hash": self.objective_hash,
            "slice_hash": self.slice_hash,
            "builds": self.builds,
            "nodes": [asdict(item) for item in self.nodes.values()],
            "edges": [asdict(item) for item in self.edges],
            "failure_kinds": self.failure_kinds,
            "ownership": {path: list(owners) for path, owners in self.ownership.items()},
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "premortem": [asdict(item) for item in self.premortem],
        }

    def _add_node(self, path: str, kind: str, reason: str) -> None:
        if not path or (path not in self.nodes and len(self.nodes) >= MAX_NODES):
            return
        current = self.nodes.get(path)
        kinds = tuple(dict.fromkeys([*(current.kinds if current else ()), kind]))[:6]
        self.nodes[path] = WorldNode(path, kinds, (current.reason if current else reason)[:160])

    def _add_edge(self, source: str, target: str, relation: str) -> None:
        edge = WorldEdge(source, target, relation)
        if source and target and edge not in self.edges and len(self.edges) < MAX_EDGES:
            self.edges.append(edge)

    @staticmethod
    def _slice_fingerprint(
        task: str, targets: Iterable[str], changed_paths: Iterable[str], edit_generation: int
    ) -> str:
        objective = hashlib.sha256(task.encode()).hexdigest()[:16] if task else ""
        return hashlib.sha256(
            json.dumps(
                [
                    objective,
                    sorted(str(value) for value in targets),
                    sorted(str(value) for value in changed_paths),
                    edit_generation,
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]

    def needs_prepare(
        self, task: str, targets: Iterable[str], changed_paths: Iterable[str], edit_generation: int
    ) -> bool:
        """Avoid repository and history reads when the lazy slice is unchanged."""
        return self.slice_hash != self._slice_fingerprint(
            task, targets, changed_paths, edit_generation
        )

    @staticmethod
    def _nearby_tests(root: Path, source: str) -> list[str]:
        path = root / source
        stem = path.stem.removeprefix("test_").removesuffix("_test")
        candidates = [
            path.with_name(f"test_{stem}{path.suffix}"),
            path.with_name(f"{stem}_test{path.suffix}"),
            root / "tests" / f"test_{stem}{path.suffix}",
        ]
        return [
            rel
            for candidate in candidates
            if candidate.is_file() and (rel := _relative(root, candidate))
        ]

    @staticmethod
    def _imports(path: Path) -> list[str]:
        try:
            if path.suffix not in _SOURCE_SUFFIXES or path.stat().st_size > MAX_SOURCE_BYTES:
                return []
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        values: list[str] = []
        for pattern in _IMPORT_PATTERNS:
            for match in pattern.finditer(text):
                value = next((group for group in match.groups() if group), "")
                if value and value not in values:
                    values.append(value[:200])
        return values[:24]

    @staticmethod
    def _codeowners(root: Path) -> list[tuple[str, tuple[str, ...]]]:
        source = next(
            (
                path
                for path in (
                    root / ".github" / "CODEOWNERS",
                    root / "CODEOWNERS",
                    root / "docs" / "CODEOWNERS",
                )
                if path.is_file()
            ),
            None,
        )
        if source is None:
            return []
        try:
            if source.stat().st_size > MAX_SOURCE_BYTES:
                return []
            lines = source.read_text(encoding="utf-8", errors="replace").splitlines()[:500]
        except OSError:
            return []
        rules: list[tuple[str, tuple[str, ...]]] = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2 and not parts[0].startswith("#"):
                rules.append((parts[0].lstrip("/"), tuple(parts[1:13])))
        return rules

    @staticmethod
    def _owners(path: str, rules: list[tuple[str, tuple[str, ...]]]) -> tuple[str, ...]:
        owners: tuple[str, ...] = ()
        for pattern, values in rules:
            normalized = pattern.rstrip("/")
            if fnmatch(path, normalized) or fnmatch(path, f"{normalized}/**"):
                owners = values
        return owners

    @staticmethod
    def _resolve_import(root: Path, source: Path, name: str) -> str | None:
        if name.startswith("."):
            base = source.parent / name.replace(".", "/")
        else:
            normalized = name.replace("::", "/").replace(".", "/")
            base = root / normalized
        for candidate in (
            base,
            *[base.with_suffix(s) for s in _SOURCE_SUFFIXES],
            base / "__init__.py",
        ):
            if candidate.is_file():
                return _relative(root, candidate)
        return None

    def prepare(
        self,
        *,
        task: str,
        facts: ProjectFacts,
        targets: Iterable[str],
        changed_paths: Iterable[str],
        assessment: MetacognitiveAssessment,
        edit_generation: int,
        failure_drafts: Iterable[dict[str, Any]] = (),
    ) -> None:
        targets = list(targets)
        changed_paths = list(changed_paths)
        failure_drafts = list(failure_drafts)
        fingerprint = hashlib.sha256(task.encode()).hexdigest()[:16] if task else ""
        if fingerprint != self.objective_hash:
            self.objective_hash = fingerprint
            self.nodes.clear()
            self.edges.clear()
            self.prediction = None
            self.premortem = []
            self.slice_hash = ""
            self.ownership = {}
        # Direct information tasks do not pay repository-analysis overhead.
        if assessment.execution_mode == "direct" and not changed_paths:
            return
        root = Path(facts.root)
        slice_hash = self._slice_fingerprint(task, targets, changed_paths, edit_generation)
        if slice_hash == self.slice_hash:
            return
        self.slice_hash = slice_hash
        seeds: list[str] = []
        for manifest in facts.manifests:
            self._add_node(manifest, "manifest", "detected project manifest")
            seeds.append(manifest)
        for path in instruction_files(root, targets)[:MAX_SEEDS]:
            if relative := _relative(root, path):
                self._add_node(relative, "instruction", "applicable repository instruction")
        for value in [*targets, *changed_paths]:
            relative = _relative(root, value)
            if relative and (root / relative).is_file():
                kind = "changed" if value in changed_paths else "target"
                self._add_node(relative, kind, f"current {kind} path")
                seeds.append(relative)
        for source in list(dict.fromkeys(seeds))[:MAX_SEEDS]:
            path = root / source
            for imported in self._imports(path):
                target = self._resolve_import(root, path, imported)
                if target:
                    self._add_node(target, "dependency", f"imported by {source}")
                    self._add_edge(source, target, "imports")
            for test in self._nearby_tests(root, source):
                self._add_node(test, "test", f"nearby test for {source}")
                self._add_edge(test, source, "tests")
        owner_rules = self._codeowners(root)
        for path in list(self.nodes):
            owners = self._owners(path, owner_rules)
            if owners:
                self.ownership[path] = owners
        self.failure_kinds = {}
        for draft in failure_drafts[:200]:
            kind = str(draft.get("failure_kind", "other"))[:40]
            self.failure_kinds[kind] = self.failure_kinds.get(kind, 0) + 1
        self.builds += 1
        if self.prediction is None or self.prediction.edit_generation == edit_generation:
            self.prediction = self._predict(task, facts, assessment)
            self.prediction.edit_generation = max(0, edit_generation)
        if assessment.risk_score >= 6 and self.prediction.edit_generation == edit_generation:
            self.premortem = self._premortem(facts, assessment)

    def observe_paths(self, root_value: str, paths: Iterable[str], relation: str) -> None:
        root = Path(root_value)
        for value in list(paths)[:20]:
            relative = _relative(root, value)
            if not relative:
                continue
            kind = "semantic" if relation == "semantic" else "observed"
            self._add_node(relative, kind, f"observed by {relation}")

    def _predict(
        self, task: str, facts: ProjectFacts, assessment: MetacognitiveAssessment
    ) -> ImpactPrediction:
        terms = _terms(task)
        scored: list[tuple[int, str]] = []
        for path, node in self.nodes.items():
            words = set(re.findall(r"[a-z0-9_]+", path.lower()))
            score = len(words & terms) * 3
            score += 3 if "changed" in node.kinds or "target" in node.kinds else 0
            score += 1 if "dependency" in node.kinds else 0
            if score:
                scored.append((score, path))
        files = [path for _, path in sorted(scored, key=lambda item: (-item[0], item[1]))[:20]]
        tests = [path for path, node in self.nodes.items() if "test" in node.kinds][:20]
        packaging = [
            path
            for path in self.nodes
            if Path(path).name in _PACKAGE_FILES
            or path.startswith(("registry/", "scripts/", ".github/"))
        ][:20]
        lowered = task.lower()
        platforms = sorted({label for word, label in _PLATFORM_WORDS.items() if word in lowered})
        if assessment.task_family == "operations" and not platforms:
            platforms = ["Linux", "macOS", "Windows"]
        reasons = ["task terms matched bounded repository metadata"]
        if packaging:
            reasons.append("release or dependency metadata may be affected")
        if self.failure_kinds:
            reasons.append("historical failure classes increase verification priority")
        return ImpactPrediction(
            objective_hash=self.objective_hash,
            edit_generation=assessment.risk_score * 0,  # replaced by caller generation below
            files=files,
            tests=tests,
            packaging=packaging,
            platforms=platforms,
            reasons=reasons,
        )

    def compare(self, root_value: str, changed_paths: Iterable[str], checks: Iterable[str]) -> None:
        if self.prediction is None:
            return
        root = Path(root_value)
        observed = [rel for value in changed_paths if (rel := _relative(root, value))]
        predicted = set(self.prediction.files) | set(self.prediction.packaging)
        self.prediction.observed_files = list(dict.fromkeys(observed))[:40]
        self.prediction.observed_checks = [str(value)[:120] for value in checks][:20]
        self.prediction.unexpected_files = [value for value in observed if value not in predicted][
            :20
        ]
        self.prediction.compared = True

    def _premortem(
        self, facts: ProjectFacts, assessment: MetacognitiveAssessment
    ) -> list[PremortemItem]:
        items = [
            PremortemItem(
                "A predicted or imported caller is missed by the edit.",
                "Inspect semantic references and run the narrowest affected tests.",
            ),
            PremortemItem(
                "The implementation passes stale pre-edit evidence.",
                "Require a passing check from the current edit generation.",
            ),
        ]
        prediction = self.prediction
        if prediction and prediction.packaging:
            items.append(
                PremortemItem(
                    "Package, registry, or installer metadata diverges.",
                    "Build distributions and validate version-pinned installation metadata.",
                )
            )
        if prediction and len(prediction.platforms) > 1:
            items.append(
                PremortemItem(
                    "Behavior succeeds locally but breaks a supported platform.",
                    "Run platform-aware tests or require cross-platform CI evidence.",
                )
            )
        if facts.dirty:
            items.append(
                PremortemItem(
                    "Existing user changes are overwritten or mistaken for agent work.",
                    "Compare the pre-edit status and review the final scoped diff.",
                )
            )
        if assessment.uncertainties:
            items.append(
                PremortemItem(
                    "A material uncertainty remains hidden behind a plausible completion claim.",
                    "Resolve or explicitly report each high-severity uncertainty with "
                    "fresh evidence.",
                )
            )
        return items[:MAX_PREMORTEM_ITEMS]

    def model_context(self) -> str:
        if not self.nodes and not self.prediction:
            return ""
        payload = {
            "world": {
                "nodes": [asdict(value) for value in list(self.nodes.values())[:40]],
                "edges": [asdict(value) for value in self.edges[:60]],
                "historical_failure_classes": self.failure_kinds,
                "ownership": {
                    path: list(owners) for path, owners in list(self.ownership.items())[:40]
                },
            },
            "impact_prediction": self.prediction.to_dict() if self.prediction else None,
            "pre_mortem": [asdict(value) for value in self.premortem],
            "constraints": (
                "Treat this as bounded advisory metadata. Confirm material impact with fresh "
                "tools; do not expand permissions or claim unobserved coverage."
            ),
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))[:12_000]
        return safe_context_text(encoded, "repository-intelligence")

    def render(self) -> str:
        lines = ["🗺️ **Repository Intelligence**"]
        lines.append(f"- Lazy world: {len(self.nodes)} nodes · {len(self.edges)} edges")
        if self.failure_kinds:
            lines.append(
                "- Historical failure classes: "
                + ", ".join(f"{key}={value}" for key, value in sorted(self.failure_kinds.items()))
            )
        prediction = self.prediction
        if prediction:
            lines.extend(
                (
                    f"- Predicted files: {', '.join(prediction.files) or 'none yet'}",
                    f"- Predicted tests: {', '.join(prediction.tests) or 'none yet'}",
                    "- Packaging/platforms: "
                    + (", ".join(prediction.packaging + prediction.platforms) or "none"),
                )
            )
            if prediction.compared:
                lines.append(
                    f"- Observed comparison: {len(prediction.observed_files)} files; "
                    f"{len(prediction.unexpected_files)} unexpected"
                )
        if self.premortem:
            lines.append("\n**Counterfactual pre-mortem**")
            lines.extend(f"- {item.failure} Detection: {item.detection}" for item in self.premortem)
        return "\n".join(lines)[:12_000]
