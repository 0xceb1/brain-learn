import os
from dotenv import load_dotenv
import requests
import sys
from time import time
from src.logger import Logger


def create_session(logger, username: str, password: str):
    """Create and authenticate a new session."""
    s = requests.Session()
    s.auth = (username, password)

    # Send a POST request to the /authentication API
    response = s.post('https://api.worldquantbrain.com/authentication')

    if response.status_code == 201:
        logger.log('Authentication successful.')
        return s
    else:
        logger.error('Failed to authenticate.')
        logger.error(f'Status Code: {response.status_code}')
        logger.error(f'Response: {response.text}')
        return None


def main():
    logger = Logger(
        job_name='brain-learn',
        console_log=True,
        file_log=True,
        logs_directory='logs',
        incremental_run_number=True,
    )

    load_dotenv()
    username = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')
    if not username or not password:
        logger.error('USERNAME or PASSWORD environment variables not set.')
        logger.error('Please check your .env file.')
        sys.exit(1)

    # Create and authenticate the session
    s = create_session(logger, username, password)
    if not s:
        logger.error('Exiting due to authentication failure.')
        return

    from src.genetic import GPLearnSimulator

    simulator = GPLearnSimulator(
        s,
        logger,
        username,
        password,
        population_size=100,
        generations=50,
        tournament_size=5,
        p_crossover=0.6,
        p_mutation=0.15,
        p_subtree_mutation=0.1,
        parsimony_coefficient=0.02,
        random_state=int(time() / 1000),
        # init_population=...
        max_depth=5,
        max_operators=6,
    )
    simulator.evolve()


if __name__ == '__main__':
    main()
