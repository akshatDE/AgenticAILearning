"""Arithmetic, without handing the model an interpreter.

The expression arrives from the model, which is influenced by user input and
— through ``search_internet`` — by the contents of web pages. ``eval()`` on
that path is arbitrary code execution, so the expression is parsed and walked
instead: numbers and arithmetic operators only, no names, no calls.
"""

from __future__ import annotations

import ast
import operator

from fastmcp import FastMCP

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPONENT = 1_000  # keeps 9**9**9 from freezing the server


def evaluate(node: ast.AST) -> float:
    """Walk one AST node, permitting only arithmetic."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        left, right = evaluate(node.left), evaluate(node.right)
        if op is operator.pow and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent is too large")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        return op(evaluate(node.operand))

    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, returning the result or the reason it failed."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
    except ZeroDivisionError:
        return "Cannot divide by zero."
    except (SyntaxError, ValueError) as exc:
        return f"Could not work that out: {exc}"
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the model
        return f"Could not work that out: {exc}"

    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def register(mcp: FastMCP) -> None:
    """Attach this tool to the server."""

    @mcp.tool
    def calculator(expression: str) -> str:
        """Work out an arithmetic expression, e.g. (145 * 32) + 78.

        Supports + - * / // % ** and parentheses. Numbers only — no variables,
        function calls, or names.
        """
        return calculate(expression)
