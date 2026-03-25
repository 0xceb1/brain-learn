import requests
from functools import partial
from typing import Callable

from sympy import Integer as Int

from src.brain import AlphaPerf, simulate


def evaluate_fitness(
    s: requests.Session, logger=None
) -> Callable[[str], AlphaPerf | None]:
    return partial(simulate, s, logger=logger)


def help_check_same(x: Int, y: Int) -> Int:
    if x == y:
        return x
    else:
        raise ValueError('This binary operator requires the same unit for both inputs')
