from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
PKG_ROOT = SRC_ROOT / "insurance_pricing"

CANONICAL_DIRS = (
    "data",
    "features",
    "cv",
    "models",
    "training",
    "evaluation",
    "inference",
    "analytics",
    "runtime",
)


def _module_name(py_path: Path) -> str:
    rel = py_path.relative_to(SRC_ROOT).with_suffix("")
    return "src." + ".".join(rel.parts)


def _discover_modules() -> Dict[str, Path]:
    modules: Dict[str, Path] = {}
    for py_path in PKG_ROOT.rglob("*.py"):
        modules[_module_name(py_path)] = py_path
    return modules


def _imported_modules(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def _build_graph(modules: Dict[str, Path]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    module_names = set(modules)

    for mod, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported in _imported_modules(tree):
            if imported in module_names:
                graph[mod].add(imported)
    return graph


def _detect_cycles(graph: Dict[str, Set[str]], nodes: Iterable[str]) -> List[List[str]]:
    white, gray, black = 0, 1, 2
    state = {n: white for n in nodes}
    stack: List[str] = []
    cycles: List[List[str]] = []

    def dfs(node: str) -> None:
        state[node] = gray
        stack.append(node)
        for nxt in graph.get(node, ()):
            if nxt not in state:
                continue
            if state[nxt] == white:
                dfs(nxt)
            elif state[nxt] == gray:
                i = stack.index(nxt)
                cycles.append(stack[i:] + [nxt])
        stack.pop()
        state[node] = black

    for n in sorted(nodes):
        if state[n] == white:
            dfs(n)
    return cycles


def _canonical_modules(modules: Dict[str, Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for mod, path in modules.items():
        rel = path.relative_to(PKG_ROOT)
        if len(rel.parts) < 2:
            continue
        top = rel.parts[0]
        if top in CANONICAL_DIRS:
            out[mod] = path
    return out


def _legacy_import_issues(canonical: Dict[str, Path]) -> List[str]:
    issues: List[str] = []
    for mod, path in canonical.items():
        if mod == "src.insurance_pricing.training" or mod.startswith("src.insurance_pricing.training."):
            # One-cycle compatibility exception for training module.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported in _imported_modules(tree):
            if imported == "src.insurance_pricing.legacy" or imported.startswith(
                "src.insurance_pricing.legacy."
            ):
                issues.append(f"Canonical module imports legacy: {mod} -> {imported}")
    return issues


def analyze() -> Dict[str, object]:
    modules = _discover_modules()
    graph = _build_graph(modules)
    canonical = _canonical_modules(modules)
    canonical_mod_names = list(canonical.keys())

    cycles = _detect_cycles(graph, canonical_mod_names)
    cycle_issues = [" -> ".join(c) for c in cycles]
    legacy_issues = _legacy_import_issues(canonical)

    issues = []
    issues.extend([f"Import cycle detected: {c}" for c in cycle_issues])
    issues.extend(legacy_issues)

    return {
        "status": "ok" if not issues else "failed",
        "issues": issues,
        "canonical_module_count": len(canonical_mod_names),
        "cycle_count": len(cycles),
    }


def main() -> None:
    report = analyze()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
