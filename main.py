import os
from dotenv import load_dotenv
import requests
import sys
from src.function import *
from time import time
import dill
from src.logger import Logger


def create_session(logger):
    """Create and authenticate a new session."""
    # Load credentials from .env file
    load_dotenv()
    username = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')

    if not username or not password:
        logger.error('USERNAME or PASSWORD environment variables not set.')
        logger.error('Please check your .env file.')
        sys.exit(1)

    # Create a new session
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
    # Initialize logger
    logger = Logger(
        job_name='brain-learn',
        console_log=True,
        file_log=True,
        logs_directory='logs',
        incremental_run_number=True,
    )

    # Create and authenticate the session
    s = create_session(logger)
    if not s:
        logger.error('Exiting due to authentication failure.')
        return

    _INIT_POP_LIST = dill.load(open('initial-population.pkl', 'rb'))

    # Run the GPLearn simulator
    from src.genetic import GPLearnSimulator

    simulator = GPLearnSimulator(
        session=s,
        population_size=100,
        generations=50,
        tournament_size=5,
        p_crossover=0.6,
        p_mutation=0.15,
        p_subtree_mutation=0.1,
        parsimony_coefficient=0.02,
        random_state=int(time() / 1000),
        # init_population = INIT_POP_LIST
        max_depth=5,
        max_operators=6,
        logger=logger,
    )
    simulator.evolve()


if __name__ == '__main__':
    main()
