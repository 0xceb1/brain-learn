from typing import Callable
from sympy import Integer as Int


def help_check_same(x: Int, y: Int) -> Int:
    if x == y:
        return x
    else:
        raise ValueError('This binary operator requires the same unit for both inputs')


class Terminal:
    def __init__(self, name: str, unit: Int, weight: float = 1.0):
        self.name = name
        self.unit = unit
        self.weight = weight

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Terminal({self.name})'

    def __eq__(self, other):
        return self.name == other.name


class Operator:
    def __init__(
        self,
        name: str,
        arity: int,
        function: Callable[..., str],
        unit_rule: Callable[..., Int],
        weight: float = 1.0,
    ):
        self.name = name
        self.arity = arity
        self.function = function
        self.unit_rule = unit_rule
        self.weight = weight

    def __str__(self):
        return self.name

    def __call__(self, *args):
        return self.function(*args)

    def __repr__(self):
        return f'Operator({self.name}, arity={self.arity})'

    def __eq__(self, other):
        return self.name == other.name


# all terminals
OPEN = Terminal('open', Int(3), 0.6)
CLOSE = Terminal('close', Int(3), 0.9)
HIGH = Terminal('high', Int(3), 0.6)
LOW = Terminal('low', Int(3), 0.6)
VWAP = Terminal('vwap', Int(3), 1.0)

VOLUME = Terminal('volume', Int(2), 1.2)
ADV = Terminal('ts_mean(volume, 63)', Int(2), 1.2)

VOL = Terminal('ts_std_dev(returns, 63)', Int(1), 1.2)
RET = Terminal('returns', Int(1), 1.2)

MOMENTUM = Terminal('(close/ts_delay(close, 180)-1)', Int(1), 0.5)
REVERSAL = Terminal('(1-close/ts_delay(close, 5))', Int(1), 1.0)
BETA = Terminal('beta_last_60_days_spy', Int(1), 0.5)
CORR_SPY = Terminal('ts_corr(close,spy,63)', Int(1), 0.5)
SYS_RISK = Terminal('systematic_risk_last_60_days', Int(1), 0.5)
IDIO_RISK = Terminal('unsystematic_risk_last_60_days', Int(1), 0.5)
CORR_DIV = Terminal(
    'abs(correlation_last_30_days_spy-correlation_last_360_days_spy)', Int(1), 1.0
)

# price volume divergence
PV_DIV1 = Terminal('-ts_corr(ts_rank(volume,10),ts_rank(close,10),10)', Int(1), 0.8)
PV_DIV2 = Terminal('-ts_corr(ts_delta(volume,1),ts_delta(close,1),10)', Int(1), 0.8)
PV_DIV3 = Terminal(
    '-ts_corr(rank(ts_delta(volume,1)),rank(ts_delta(close,1)),10)', Int(1), 0.5
)

# here are some factors that I can't give a good name :(
FACTOR_1 = Terminal('ts_corr(close,open,10)', Int(1), 1.0)
FACTOR_2 = Terminal('(high+low)/2-close', Int(3), 0.5)
FACTOR_3 = Terminal('abs(close-vwap)', Int(3), 0.5)

# technical indicators
BOLL = Terminal('-ts_zscore(close, 21)', Int(1), 1.2)
WILLIAMS = Terminal(
    '((ts_mean(high, 10)-close)/(ts_mean(high,10)-ts_mean(low,10)))', Int(1), 1.2
)

ONE = Terminal('1', Int(1), 0)
ZERO = Terminal('0', Int(1), 0)

# all operators
ADD = Operator(
    'add',
    2,
    lambda x, y: f'add({x},{y})',
    lambda x, y: help_check_same(x, y),
    weight=2.0,
)
SUB = Operator(
    'sub',
    2,
    lambda x, y: f'subtract({x},{y})',
    lambda x, y: help_check_same(x, y),
    weight=2.0,
)
MUL = Operator(
    'mul', 2, lambda x, y: f'multiply({x},{y})', lambda x, y: x * y, weight=1.5
)  # better not to use MUL on terminal nodes
DIV = Operator(
    'div', 2, lambda x, y: f'divide({x},{y})', lambda x, y: x * y, weight=1.5
)
MAX = Operator(
    'max',
    2,
    lambda x, y: f'max({x},{y})',
    lambda x, y: help_check_same(x, y),
    weight=1.5,
)
MIN = Operator(
    'min',
    2,
    lambda x, y: f'min({x},{y})',
    lambda x, y: help_check_same(x, y),
    weight=1.5,
)
REV = Operator('rev', 1, lambda x: f'reverse({x})', lambda x: x, weight=0.2)
# SQRT = Operator('sqrt', 1, lambda x: f'sqrt({x})', lambda x: sp.sqrt(x), weight=0.1)
INV = Operator('inv', 1, lambda x: f'inverse({x})', lambda x: 1 / x, weight=0.2)
# SQUARE = Operator('square', 1, lambda x: f'power({x}, 2)',lambda x: x*x, weight=0.1)
ABS = Operator('abs', 1, lambda x: f'abs({x})', lambda x: x, weight=0.5)
SIGN = Operator('sign', 1, lambda x: f'sign({x})', lambda _: 1, weight=0.5)

CORR_10 = Operator(
    'corr10', 2, lambda x, y: f'ts_corr({x},{y},10)', lambda _, _unused: 1, weight=0.8
)
CORR_21 = Operator(
    'corr21', 2, lambda x, y: f'ts_corr({x},{y},21)', lambda _, _unused: 1, weight=0.8
)
CORR_63 = Operator(
    'corr63', 2, lambda x, y: f'ts_corr({x},{y},63)', lambda _, _unused: 1, weight=0.8
)
RCORR_10 = Operator(
    'RCORR10',
    2,
    lambda x, y: f'ts_corr(ts_rank({x},10),ts_rank({y},10),10)',
    lambda _, _unused: 1,
    weight=0.5,
)
RCORR_21 = Operator(
    'RCORR21',
    2,
    lambda x, y: f'ts_corr(ts_rank({x},21),ts_rank({y},21),21)',
    lambda _, _unused: 1,
    weight=0.5,
)
RCORR_63 = Operator(
    'RCORR63',
    2,
    lambda x, y: f'ts_corr(ts_rank({x},63),ts_rank({y},63),63)',
    lambda _, _unused: 1,
    weight=0.5,
)

TSR_10 = Operator('tsr10', 1, lambda x: f'ts_rank({x},10)', lambda _: 1, weight=0.5)
TSR_21 = Operator('tsr21', 1, lambda x: f'ts_rank({x},21)', lambda _: 1, weight=0.5)
TSR_63 = Operator('tsr63', 1, lambda x: f'ts_rank({x},63)', lambda _: 1, weight=0.5)

TSS_10 = Operator('tss10', 1, lambda x: f'ts_sum({x},10)', lambda _: 1, weight=0.5)
TSS_21 = Operator('tss21', 1, lambda x: f'ts_sum({x},21)', lambda _: 1, weight=0.5)
TSS_63 = Operator('tss63', 1, lambda x: f'ts_sum({x},63)', lambda _: 1, weight=0.5)

TSZ_21 = Operator('tsz21', 1, lambda x: f'ts_zscore({x},21)', lambda _: 1, weight=0.8)
TSZ_63 = Operator('tsz63', 1, lambda x: f'ts_zscore({x},63)', lambda _: 1, weight=0.8)

STD_10 = Operator('std10', 1, lambda x: f'ts_std_dev({x},10)', lambda _: 1, weight=0.5)
STD_21 = Operator('std21', 1, lambda x: f'ts_std_dev({x},21)', lambda _: 1, weight=0.5)
STD_63 = Operator('std63', 1, lambda x: f'ts_std_dev({x},63)', lambda _: 1, weight=0.5)

SURPRISE_63 = Operator(
    'surprise63', 1, lambda x: f'({x}/ts_mean({x}, 63))', lambda _: 1, weight=0.5
)
SURPRISE_21 = Operator(
    'surprise21', 1, lambda x: f'({x}/ts_mean({x}, 21))', lambda _: 1, weight=0.5
)
SURPRISE_10 = Operator(
    'surprise10', 1, lambda x: f'({x}/ts_mean({x}, 10))', lambda _: 1, weight=0.5
)

REG_RESD_63 = Operator(
    'reg_residuals63',
    2,
    lambda x, y: f'ts_regression({x},{y},63,lag=0,rettype=0)',
    lambda _, y: y,
    weight=1.2,
)
REG_COEF_63 = Operator(
    'reg_coefficient63',
    2,
    lambda x, y: f'ts_regression({x},{y},63,lag=0,rettype=2)',
    lambda _, _unused: 1,
    weight=1.2,
)

RANK = Operator('rank', 1, lambda x: f'rank({x})', lambda _: 1, weight=2)
ZSCORE = Operator('zscore', 1, lambda x: f'zscore({x})', lambda _: 1, weight=2)
WINSORIZE = Operator(
    'winsorize', 1, lambda x: f'winsorize({x},std=5)', lambda _: 1, weight=1
)

ARITIES = {
    2: [
        ADD,
        SUB,
        MUL,
        DIV,
        MAX,
        MIN,
        CORR_10,
        CORR_21,
        CORR_63,
        RCORR_10,
        RCORR_21,
        RCORR_63,
        REG_RESD_63,
        REG_COEF_63,
    ],
    1: [
        INV,
        REV,
        ABS,
        SIGN,
        RANK,
        ZSCORE,
        WINSORIZE,
        TSR_10,
        TSR_21,
        TSR_63,
        TSS_10,
        TSS_21,
        TSS_63,
        TSZ_21,
        TSZ_63,
        STD_10,
        STD_21,
        STD_63,
        SURPRISE_10,
        SURPRISE_21,
        SURPRISE_63,
    ],
}

OPERATORS = [
    # Binary operators
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MIN,
    CORR_10,
    CORR_21,
    CORR_63,
    RCORR_10,
    RCORR_21,
    RCORR_63,
    REG_RESD_63,
    REG_COEF_63,
    # Unary operators
    INV,
    REV,
    ABS,
    SIGN,
    RANK,
    ZSCORE,
    WINSORIZE,
    TSR_10,
    TSR_21,
    TSR_63,
    TSS_10,
    TSS_21,
    TSS_63,
    TSZ_21,
    TSZ_63,
    STD_10,
    STD_21,
    STD_63,
    SURPRISE_10,
    SURPRISE_21,
    SURPRISE_63,
]
TERMINALS = [
    OPEN,
    CLOSE,
    HIGH,
    LOW,
    VWAP,
    VOLUME,
    ADV,
    RET,
    MOMENTUM,
    REVERSAL,
    BETA,
    ONE,
    ZERO,
    FACTOR_1,
    PV_DIV1,
    PV_DIV2,
    PV_DIV3,
    FACTOR_2,
    FACTOR_3,
    BOLL,
    WILLIAMS,
    CORR_SPY,
    SYS_RISK,
    IDIO_RISK,
    CORR_DIV,
]
