"""
AST guard: verify generated strategy code is safe.
"""

from __future__ import annotations

import ast
DANGEROUS_NAMES = {
    "os", "sys", "subprocess", "socket", "requests", "urllib",
    "exec", "eval", "compile", "__import__", "open", "input",
    "getattr", "setattr", "delattr", "globals", "locals",
}


def ast_guard(code: str) -> list[str]:
    """Return list of violations. Empty = safe."""
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DANGEROUS_NAMES:
                    violations.append(f"Blocked import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DANGEROUS_NAMES:
                violations.append(f"Blocked import from: {node.module}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile"):
                violations.append(f"Blocked call: {node.func.id}()")

    return violations
