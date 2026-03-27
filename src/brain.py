import requests
from enum import StrEnum
from functools import partial
from time import sleep
import pandas as pd
from datetime import datetime
import os
import threading
from typing import Callable, TypedDict, cast

from src.logger import Logger


class PassOrFail(StrEnum):
    PASS = 'PASS'
    FAIL = 'FAIL'
    PENDING = 'PENDING'


def _check_fields(cr: dict[str, str]) -> dict[str, PassOrFail | None]:
    pairs = (
        ('LOW_SHARPE', 'low_sharpe'),
        ('LOW_FITNESS', 'low_fitness'),
        ('LOW_TURNOVER', 'low_turnover'),
        ('HIGH_TURNOVER', 'high_turnover'),
        ('CONCENTRATED_WEIGHT', 'concentrated_weight'),
        ('LOW_SUB_UNIVERSE_SHARPE', 'low_sub_universe_sharpe'),
        ('SELF_CORRELATION', 'self_correlation'),
        ('MATCHES_COMPETITION', 'matches_competition'),
    )
    out: dict[str, PassOrFail | None] = {}
    for api, key in pairs:
        v = cr.get(api)
        out[key] = PassOrFail(v) if v in PassOrFail else None
    return out


class AlphaPerf(TypedDict, total=False):
    alpha_id: str
    regular_code: str | None
    turnover: float | None
    returns: float | None
    drawdown: float | None
    margin: float | None
    fitness: float | None
    sharpe: float | None
    low_sharpe: PassOrFail | None
    low_fitness: PassOrFail | None
    low_turnover: PassOrFail | None
    high_turnover: PassOrFail | None
    concentrated_weight: PassOrFail | None
    low_sub_universe_sharpe: PassOrFail | None
    self_correlation: PassOrFail | None
    matches_competition: PassOrFail | None
    expression: str
    final_fitness: float
    error: str


_csv_lock = threading.Lock()


def save_alpha(alpha_performance: AlphaPerf, logger: Logger):
    if not alpha_performance:
        return

    csv_path = 'simulation_results.csv'
    perf_to_save = cast(dict, alpha_performance.copy())
    perf_to_save['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        df_new = pd.DataFrame([perf_to_save])
        with _csv_lock:
            if os.path.exists(csv_path):
                df_new.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df_new.to_csv(csv_path, index=False)
    except Exception as e:
        logger.error(f'Failed to write to CSV file: {e}')


def read_simulations_csv(csv_path='simulation_results.csv'):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        error_msg = f"Error reading or processing CSV file '{csv_path}': {e}"
        print(f' Error: {error_msg}')
        return pd.DataFrame()


def get_alpha_performance(s: requests.Session, alpha_id: str) -> AlphaPerf:
    alpha = s.get('https://api.worldquantbrain.com/alphas/' + alpha_id)
    regular = alpha.json().get('regular', {})
    investment_summary = alpha.json().get('is', {})
    checks = alpha.json().get('is', {}).get('checks', [])
    check_results = {check['name']: check['result'] for check in checks}

    alpha_performance = {
        'alpha_id': alpha_id,
        'regular_code': regular.get('code'),
        'turnover': investment_summary.get('turnover'),
        'returns': investment_summary.get('returns'),
        'drawdown': investment_summary.get('drawdown'),
        'margin': investment_summary.get('margin'),
        'fitness': investment_summary.get('fitness'),
        'sharpe': investment_summary.get('sharpe'),
        **_check_fields(check_results),
    }
    return cast(AlphaPerf, alpha_performance)


def simulate(
    s: requests.Session, logger: Logger, fast_expr: str, timeout=300
) -> AlphaPerf | None:
    simulation_data = {
        'type': 'REGULAR',
        'settings': {
            'instrumentType': 'EQUITY',
            'region': 'USA',
            'universe': 'TOP3000',
            'delay': 1,
            'decay': 1,
            'neutralization': 'INDUSTRY',
            'truncation': 0.1,
            'pasteurization': 'ON',
            'unitHandling': 'VERIFY',
            'nanHandling': 'OFF',
            'language': 'FASTEXPR',
            'visualization': False,
        },
        'regular': fast_expr,
    }

    # Maximum number of retries for rate limiting
    MAX_RETRIES = 5

    for retry in range(MAX_RETRIES):
        simulation_response = s.post(
            'https://api.worldquantbrain.com/simulations', json=simulation_data
        )

        if simulation_response.status_code == 429:
            error_message = simulation_response.text
            if 'SIMULATION_LIMIT_EXCEEDED' in error_message:
                wait_time = 2 ** (retry + 1)  # 1, 2, 4, 8, 16 seconds
                logger.warning(
                    f'Rate limit exceeded. Waiting {wait_time} seconds before retry {retry + 1}/{MAX_RETRIES}...'
                )
                sleep(wait_time)
                continue
            else:
                logger.error(
                    f'Failed to send simulation. Status code: {simulation_response.status_code}.'
                )
                logger.error(f'Response: {simulation_response.text}')
                return None

        if simulation_response.status_code == 401:
            logger.error('Authentication error: Incorrect credentials.')
            logger.error(f'Response: {simulation_response.text}')
            return None

        break

    if simulation_response.status_code != 201:
        logger.error(
            f'Failed to send simulation after {MAX_RETRIES} retries. Status code: {simulation_response.status_code}.'
        )
        logger.error(f'Response: {simulation_response.text}')
        return None

    logger.log(f'Simulation sent successfully: {fast_expr}')

    simulation_progress_url = simulation_response.headers['Location']
    finished = False
    total_wait_time = 0

    while not finished and total_wait_time < timeout:
        simulation_progress = s.get(simulation_progress_url)

        if simulation_progress.status_code == 401:
            logger.error('Authentication error during simulation progress monitoring.')
            logger.error(f'Response: {simulation_progress.text}')
            return None

        if simulation_progress.headers.get('Retry-After', 0) == 0:
            finished = True
            break

        wait_time = float(simulation_progress.headers['Retry-After'])

        total_wait_time += wait_time

        if total_wait_time >= timeout:
            logger.warning(f'Timeout of {timeout} seconds will be exceeded. Aborting.')
            return None

        sleep(wait_time)

    if finished:
        try:
            alpha_id = simulation_progress.json()['alpha']
            alpha_performance = get_alpha_performance(s, alpha_id)
            if alpha_performance:
                save_alpha(alpha_performance, logger=logger)
                alpha_performance['expression'] = fast_expr
                return alpha_performance
            return None
        except Exception as e:
            logger.error(f'Error processing completed simulation: {e}')
            return None
    else:
        logger.warning(f'Simulation timed out after {total_wait_time} seconds')
        return None


def evaluate_fitness(
    s: requests.Session, logger
) -> Callable[[str], AlphaPerf | None]:
    return partial(simulate, s, logger=logger)


def get_alpha_history(s: requests.Session, pandas=True):
    all_alphas = s.get('https://api.worldquantbrain.com/users/self/alphas').json()[
        'results'
    ]
    alpha_list = []
    for alpha in all_alphas:
        regular = alpha.get('regular', {})
        investment_summary = alpha.get('is', {})
        checks = alpha.get('is', {}).get('checks', [])

        check_results = {check['name']: check['result'] for check in checks}
        data = {
            'id': alpha.get('id'),
            'regular_code': regular.get('code'),
            'turnover': investment_summary.get('turnover'),
            'returns': investment_summary.get('returns'),
            'drawdown': investment_summary.get('drawdown'),
            'margin': investment_summary.get('margin'),
            'fitness': investment_summary.get('fitness'),
            'sharpe': investment_summary.get('sharpe'),
            **_check_fields(check_results),
        }
        alpha_list.append(data)

    return pd.DataFrame(alpha_list) if pandas else alpha_list
