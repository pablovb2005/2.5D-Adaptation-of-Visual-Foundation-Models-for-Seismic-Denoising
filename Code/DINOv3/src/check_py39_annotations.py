from __future__ import annotations

import argparse
import ast
from pathlib import Path


def has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        return (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
        )
    return False


def annotation_contains_pipe(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return False
    return any(isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr) for child in ast.walk(node))


def iter_annotations(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            yield node.lineno, node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.lineno, node.returns
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if arg.annotation is not None:
                    yield arg.lineno, arg.annotation
            if node.args.vararg and node.args.vararg.annotation is not None:
                yield node.args.vararg.lineno, node.args.vararg.annotation
            if node.args.kwarg and node.args.kwarg.annotation is not None:
                yield node.args.kwarg.lineno, node.args.kwarg.annotation


def check_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    if has_future_annotations(tree):
        return []

    failures = []
    for lineno, annotation in iter_annotations(tree):
        if annotation_contains_pipe(annotation):
            failures.append(f"{path}:{lineno}: PEP 604 union annotation needs 'from __future__ import annotations'")
    return failures


def check_hubconf_imports(path: Path) -> list[str]:
    if path.name != "hubconf.py":
        return []

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    allowed = {"dinov3.hub.backbones"}
    failures = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dinov3.hub."):
            if node.module not in allowed:
                failures.append(
                    f"{path}:{node.lineno}: imports {node.module}; "
                    "keep hubconf.py backbone-only to preserve a narrow public API"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce 'from __future__ import annotations' before PEP 604 X|Y syntax, "
            "and keep hubconf.py backbone-only."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to scan")
    args = parser.parse_args()

    py_files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            py_files.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            py_files.append(path)

    failures: list[str] = []
    for path in py_files:
        failures.extend(check_file(path))
        failures.extend(check_hubconf_imports(path))

    if failures:
        print("Python 3.9 annotation preflight FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print(f"Python 3.9 annotation preflight passed ({len(py_files)} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
