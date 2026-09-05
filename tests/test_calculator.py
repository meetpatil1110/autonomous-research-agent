import math

import pytest

from agent.tools.calculator import CalculatorError, calculate


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", 4),
        ("(3 + 4) * 2", 14),
        ("2 ** 10", 1024),
        ("10 // 3", 3),
        ("sqrt(16)", 4.0),
        ("round(pi, 2)", round(math.pi, 2)),
        ("-5 + 3", -2),
    ],
)
def test_calculate_valid_expressions(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "[1, 2, 3]",
        "1 if True else 2",
        "os.system('ls')",
        "not_a_number",
    ],
)
def test_calculate_rejects_disallowed_expressions(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_calculate_rejects_syntax_errors():
    with pytest.raises(CalculatorError):
        calculate("2 +")
