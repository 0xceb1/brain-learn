import requests
from time import sleep
import pandas as pd
from datetime import datetime
import os
import threading
from functools import wraps

# Global lock for thread-safe CSV operations
_csv_lock = threading.Lock()


def thread_safe_csv(func):
    """Decorator to make any CSV operation thread-safe."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _csv_lock:
            return func(*args, **kwargs)

    return wrapper


@thread_safe_csv
def save_alpha_to_csv(alpha_performance, logger=None):
    """
    Save alpha performance data to CSV in a thread-safe manner using pandas.
    Always saves to 'simulation_results.csv' in the root directory.

    Parameters
    ----------
    alpha_performance : dict
        Dictionary containing alpha performance metrics
    logger : Logger, optional
        Logger instance for logging messages
    """
    if not alpha_performance:
        return

    csv_path = 'simulation_results.csv'

    # Add timestamp to the performance data
    alpha_performance = (
        alpha_performance.copy()
    )  # Create a copy to avoid modifying the original
    alpha_performance['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Convert single record to DataFrame
        df_new = pd.DataFrame([alpha_performance])

        # Check if file exists and append or create new
        if os.path.exists(csv_path):
            # Append to existing file without writing headers
            df_new.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # Create new file with headers
            df_new.to_csv(csv_path, index=False)

    except Exception as e:
        error_msg = f'Failed to write to CSV file: {e}'
        if logger:
            logger.error(error_msg)
        else:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {error_msg}')


@thread_safe_csv
def read_simulations_csv(csv_path='simulations.csv'):
    """
    Read simulation data from CSV in a thread-safe manner. Ensures a standard
    set of columns is present in the output DataFrame.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file (default is "simulations.csv")
    filter_criteria : dict, optional
        Dictionary of {column: value} pairs to filter the data.
        For numeric columns, filters rows where column >= value.
        For non-numeric columns, filters rows where column == value.
        Example: {'sharpe': 1.5} will return only rows where sharpe >= 1.5
    sort_by : tuple or str, optional
        Column name(s) to sort by. Can be a string (column name, descending order)
        or a tuple (column_name, ascending_bool).
        Example: 'fitness' or ('fitness', False) for descending order.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing simulation results, potentially reindexed and filtered/sorted.
        Returns an empty DataFrame with standard columns if the file doesn't exist or an error occurs.
    """
    EXPECTED_COLUMNS = [
        'alpha_id',
        'regular_code',
        'turnover',
        'returns',
        'drawdown',
        'margin',
        'fitness',
        'sharpe',
        'LOW_SHARPE',
        'LOW_FITNESS',
        'LOW_TURNOVER',
        'HIGH_TURNOVER',
        'CONCENTRATED_WEIGHT',
        'LOW_SUB_UNIVERSE_SHARPE',
        'SELF_CORRELATION',
        'MATCHES_COMPETITION',
        'datetime',
    ]

    empty_df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    try:
        if not os.path.exists(csv_path):
            print(f'Warning: CSV file not found: {csv_path}')
            return empty_df

        df = pd.read_csv(csv_path, header=None)
        df.columns = EXPECTED_COLUMNS
        df['datetime'] = pd.to_datetime(df['datetime'])

        return df

    except pd.errors.EmptyDataError:
        print(f'Warning: CSV file is empty: {csv_path}')
        return empty_df
    except Exception as e:
        error_msg = f"Error reading or processing CSV file '{csv_path}': {e}"
        print(f' Error: {error_msg}')
        # Return empty DataFrame with standard columns on error
        return empty_df


def get_alpha_performance(s: requests.Session, alpha_id: str):
    alpha = s.get('https://api.worldquantbrain.com/alphas/' + alpha_id)
    regular = alpha.json().get('regular', {})
    investment_summary = alpha.json().get('is', {})
    checks = alpha.json().get('is', {}).get('checks', [])
    check_results = {check['name']: check['result'] for check in checks}

    # 创建一个包含所需信息的字典
    alpha_performance = {
        'alpha_id': alpha_id,
        'regular_code': regular.get('code'),
        'turnover': investment_summary.get('turnover'),
        'returns': investment_summary.get('returns'),
        'drawdown': investment_summary.get('drawdown'),
        'margin': investment_summary.get('margin'),
        'fitness': investment_summary.get('fitness'),
        'sharpe': investment_summary.get('sharpe'),
        'LOW_SHARPE': check_results.get('LOW_SHARPE', 'Not Found'),
        'LOW_FITNESS': check_results.get('LOW_FITNESS', 'Not Found'),
        'LOW_TURNOVER': check_results.get('LOW_TURNOVER', 'Not Found'),
        'HIGH_TURNOVER': check_results.get('HIGH_TURNOVER', 'Not Found'),
        'CONCENTRATED_WEIGHT': check_results.get('CONCENTRATED_WEIGHT', 'Not Found'),
        'LOW_SUB_UNIVERSE_SHARPE': check_results.get(
            'LOW_SUB_UNIVERSE_SHARPE', 'Not Found'
        ),
        'SELF_CORRELATION': check_results.get('SELF_CORRELATION', 'Not Found'),
        'MATCHES_COMPETITION': check_results.get('MATCHES_COMPETITION', 'Not Found'),
    }
    return alpha_performance


def simulate(
    s: requests.Session, fast_expr: str, timeout=300, logger=None
) -> dict | None:
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
                log_msg = f'Rate limit exceeded. Waiting {wait_time} seconds before retry {retry + 1}/{MAX_RETRIES}...'
                if logger:
                    logger.warning(log_msg)
                else:
                    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg}')
                sleep(wait_time)
                continue
            else:
                log_msg1 = f'Failed to send simulation. Status code: {simulation_response.status_code}'
                log_msg2 = f'Response: {simulation_response.text}'
                if logger:
                    logger.error(log_msg1)
                    logger.error(log_msg2)
                else:
                    print(
                        f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg1}'
                    )
                    print(
                        f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg2}'
                    )
                return None

        if simulation_response.status_code == 401:
            log_msg1 = 'Authentication error: Incorrect credentials.'
            log_msg2 = f'Response: {simulation_response.text}'
            if logger:
                logger.error(log_msg1)
                logger.error(log_msg2)
            else:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg1}')
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg2}')
            return None

        break

    if simulation_response.status_code != 201:
        log_msg1 = f'Failed to send simulation after {MAX_RETRIES} retries. Status code: {simulation_response.status_code}'
        log_msg2 = f'Response: {simulation_response.text}'
        if logger:
            logger.error(log_msg1)
            logger.error(log_msg2)
        else:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg1}')
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg2}')
        return None

    if logger:
        logger.log(f'Simulation sent successfully: {fast_expr}')
    else:
        print(
            f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Simulation sent successfully: {fast_expr}'
        )

    simulation_progress_url = simulation_response.headers['Location']
    finished = False
    total_wait_time = 0

    while not finished and total_wait_time < timeout:
        simulation_progress = s.get(simulation_progress_url)

        if simulation_progress.status_code == 401:
            log_msg1 = 'Authentication error during simulation progress monitoring.'
            log_msg2 = f'Response: {simulation_progress.text}'
            if logger:
                logger.error(log_msg1)
                logger.error(log_msg2)
            else:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg1}')
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg2}')
            return None

        if simulation_progress.headers.get('Retry-After', 0) == 0:
            finished = True
            break

        wait_time = float(simulation_progress.headers['Retry-After'])

        total_wait_time += wait_time

        if total_wait_time >= timeout:
            log_msg = f'Timeout of {timeout} seconds will be exceeded. Aborting.'
            if logger:
                logger.warning(log_msg)
            else:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg}')
            return None

        sleep(wait_time)

    if finished:
        try:
            alpha_id = simulation_progress.json()['alpha']
            alpha_performance = get_alpha_performance(s, alpha_id)
            if alpha_performance:
                # Save the performance data to CSV
                save_alpha_to_csv(alpha_performance, logger=logger)
                # Add the expression to the returned alpha_performance object for reference
                alpha_performance['expression'] = fast_expr
                return alpha_performance
            return None
        except Exception as e:
            log_msg = f'Error processing completed simulation: {e}'
            if logger:
                logger.error(log_msg)
            else:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg}')
            return None
    else:
        log_msg = f'Simulation timed out after {total_wait_time} seconds'
        if logger:
            logger.warning(log_msg)
        else:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {log_msg}')
        return None


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
            'LOW_SHARPE': check_results.get('LOW_SHARPE', 'Not Found'),
            'LOW_FITNESS': check_results.get('LOW_FITNESS', 'Not Found'),
            'LOW_TURNOVER': check_results.get('LOW_TURNOVER', 'Not Found'),
            'HIGH_TURNOVER': check_results.get('HIGH_TURNOVER', 'Not Found'),
            'CONCENTRATED_WEIGHT': check_results.get(
                'CONCENTRATED_WEIGHT', 'Not Found'
            ),
            'LOW_SUB_UNIVERSE_SHARPE': check_results.get(
                'LOW_SUB_UNIVERSE_SHARPE', 'Not Found'
            ),
            'SELF_CORRELATION': check_results.get('SELF_CORRELATION', 'Not Found'),
            'MATCHES_COMPETITION': check_results.get(
                'MATCHES_COMPETITION', 'Not Found'
            ),
        }
        alpha_list.append(data)

    return pd.DataFrame(alpha_list) if pandas else alpha_list
