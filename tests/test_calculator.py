"""The calculator refuses to be an interpreter.

The expression reaches it from the model, which is influenced by user input
and by web pages fetched through search_internet. eval() on that path is
arbitrary code execution.
"""

from agenticgpt.servers.workspace.tools.calculator import calculate


def test_arithmetic():
    assert calculate("(145 * 32) + 78") == "4718"
    assert calculate("15 / 4") == "3.75"
    assert calculate("-3 + 1") == "-2"
    assert calculate("2 ** 10") == "1024"


def test_whole_floats_come_back_without_a_trailing_zero():
    assert calculate("10 / 2") == "5"


def test_division_by_zero_is_explained_not_raised():
    assert calculate("1 / 0") == "Cannot divide by zero."


def test_names_and_calls_are_refused():
    for expression in (
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "print(1)",
        "x + 1",
    ):
        assert calculate(expression).startswith("Could not work that out")


def test_huge_exponents_are_refused():
    # 9**9**9 would otherwise hang the server process.
    assert calculate("9 ** 9 ** 9").startswith("Could not work that out")


def test_syntax_errors_come_back_as_text():
    assert calculate("2 +").startswith("Could not work that out")
