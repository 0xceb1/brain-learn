import shutil
import concurrent.futures
import heapq
import os
import pickle
import threading
import time
import warnings
from typing import Callable, TypedDict, cast
from datetime import timedelta

import numpy as np
import requests

from src.function import Operator, Terminal
from src.logger import Logger
from src.program import Program
from src.brain import AlphaPerf, evaluate_fitness

# Maximum number of reconnection attempts
MAX_RECONNECTION_ATTEMPTS = 5
IS_TESTS = [
    'LOW_SHARPE',
    'LOW_FITNESS',
    'LOW_TURNOVER',
    'HIGH_TURNOVER',
    'CONCENTRATED_WEIGHT',
    'LOW_SUB_UNIVERSE_SHARPE',
    'MATCHES_COMPETITION',
]


class GenReport(TypedDict):
    generation: int
    best_fitness: float
    avg_fitness: float
    population_size: int
    hof_size: int
    best_hof_fitness: float


class GPLearnSimulator:
    session: requests.Session
    logger: Logger
    population_size: int
    generations: int
    tournament_size: int
    p_crossover: float
    p_mutation: float
    p_subtree_mutation: float
    p_hoist_mutation: float
    p_point_mutation: float
    hof_size: int
    parsimony_coefficient: float

    n_parallel: int  # max=3 for normal BRAIN account
    init_population: list[list[Operator | Terminal]] | None
    max_depth: int  # max_depth per program
    max_operators: int  # max_operators per program
    random_state: np.random.RandomState
    metric: Callable[
        [str], AlphaPerf | None
    ]  # function which evaluates the expression and return an AlphaPerf

    population: list[Program]
    history: list[GenReport]
    best_program: Program | None
    best_fitness: float
    generation: int
    # TODO: change set[str] to set[Program] for better hash performance
    evaluated_expressions: set[str]
    hall_of_fame: list[tuple[float, str, AlphaPerf]]  # (fitness, expr, performance)
    fitness_evaluations: int
    start_time: float | None
    init_population_save_path: str | os.PathLike[str] | None
    _session_lock: threading.Lock
    _auth_username: str
    _auth_password: str

    def __init__(
        self,
        session: requests.Session,
        logger: Logger,
        username: str,
        password: str,
        *,
        population_size: int = 30,
        generations: int = 20,
        tournament_size: int = 5,
        p_crossover: float = 0.7,
        p_mutation: float = 0.1,
        p_subtree_mutation: float = 0.05,
        p_hoist_mutation: float = 0.05,
        p_point_mutation: float = 0.1,
        max_depth: int = 5,
        max_operators: int = 10,
        random_state: int | np.random.RandomState = 42,
        parsimony_coefficient: float = 0.15,
        n_parallel: int = 3,
        init_population: list[list[Terminal | Operator]] | None = None,
        init_population_save_path: str
        | os.PathLike[str]
        | None = 'initial-population.pkl',
        hof_size: int = 50,
    ):
        """
        A Genetic Programming simulator optimized for simulation-based fitness evaluation.

        Parameters
        ----------
        session : requests.Session
            Authenticated BRAIN API session.
        logger : Logger
            Logger instance for logging messages.
        username : str
            BRAIN API username (same as used for ``session.auth``).
        password : str
            BRAIN API password.
        population_size : int
            Size of the population
        generations : int
            Number of generations to evolve
        tournament_size : int
            Size of tournament for selection
        p_crossover : float
            Probability of crossover
        p_mutation : float
            Probability of mutation
        p_subtree_mutation : float
            Probability of subtree mutation
        p_hoist_mutation : float
            Probability of hoist mutation
        p_point_mutation : float
            Probability of point mutation
        max_depth : int
            Maximum tree depth
        max_operators : int
            Maximum number of operators
        random_state : int or RandomState, optional
            Random number generator
        metric : callable, optional
            Function to evaluate fitness
        parsimony_coefficient : float
            Coefficient for parsimony pressure (penalty for size)
        n_parallel : int, optional
            Number of parallel workers for fitness evaluation (default is 1). Max is 3.
        init_population : list, optional
            List of Program objects to initialize the population with.
        hof_size : int, optional
            Number of best individuals to keep track of in the Hall-of-Fame (default is 50).
        """
        # Basic parameters
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.max_operators = max_operators
        self.parsimony_coefficient = parsimony_coefficient
        self.n_parallel = min(n_parallel, 3)  # Cap at 3 as per spec
        self.logger = logger
        self.hof_size = hof_size

        # Normalize crossover and mutation probabilities to sum to 1
        total_prob = p_crossover + p_mutation
        self.p_crossover = p_crossover / total_prob
        self.p_mutation = p_mutation / total_prob

        # Normalize the mutation type probabilities to sum to 1
        total_mutation = p_subtree_mutation + p_hoist_mutation + p_point_mutation
        self.p_subtree_mutation = p_subtree_mutation / total_mutation
        self.p_hoist_mutation = p_hoist_mutation / total_mutation
        self.p_point_mutation = p_point_mutation / total_mutation

        self._auth_username = username
        self._auth_password = password

        # Setup session and metric
        self.session = session
        self.metric = evaluate_fitness(session, logger=logger)

        # Setup random state
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state or np.random.RandomState()

        # Initialize tracking variables
        self.population = []
        if init_population:
            self._create_initial_programs(init_population)

        self.history = []
        self.best_program = None
        self.best_fitness = float('-inf')
        self.generation = 0
        self.evaluated_expressions = set()
        self.hall_of_fame = []

        # Statistics
        self.fitness_evaluations = 0
        self.start_time = None
        self._session_lock = threading.Lock()
        self.init_population_save_path = init_population_save_path

    def _create_initial_programs(
        self, programs: list[list[Operator | Terminal]]
    ) -> None:
        """Create Program objects from the provided initial programs."""
        self.population = [
            Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                random_state=self.random_state,
                metric=self.metric,
                parimony_coefficient=self.parsimony_coefficient,
                program=program,
            )
            for program in programs
        ]

        # Mark these as evaluated
        for prog in self.population:
            self.evaluated_expressions.add(str(prog))

    def _initialize_population(self) -> None:
        """Initialize population with random programs, avoiding duplicates."""
        # Create random programs until we reach the desired population size
        while len(self.population) < self.population_size:
            program = Program(
                max_depth=self.max_depth,
                max_operators=self.max_operators,
                random_state=self.random_state,
                metric=self.metric,
                parimony_coefficient=self.parsimony_coefficient,
            )
            program_str = str(program)

            # Only add if not seen before
            if program_str not in self.evaluated_expressions:
                self.population.append(program)
                self.evaluated_expressions.add(program_str)

        self.logger.log(f'Initialized population with {len(self.population)} programs')

    def _recreate_session(self) -> bool:
        """Recreate the session if authentication fails."""
        self.session = requests.Session()
        self.session.auth = (self._auth_username, self._auth_password)

        # Authenticate the session
        response = self.session.post('https://api.worldquantbrain.com/authentication')

        if response.status_code == 201:
            self.logger.log('Session reconnected successfully.')
            self.metric = evaluate_fitness(self.session, logger=self.logger)
            return True
        else:
            self.logger.error(
                f'Failed to reconnect session. Status Code: {response.status_code}'
            )
            self.logger.error(f'Response: {response.text}')
            return False

    def _meets_hof_threshold(self, result) -> bool:
        """Checks if a simulation result meets the criteria for Hall of Fame."""
        if not result:
            return False

        # Check if result passes all specified tests
        passes_tests = all(result.get(k) == 'PASS' for k in IS_TESTS if k in result)

        # Check threshold values (NOTE: hardcoded threshold)
        high_sharpe = result.get('sharpe', -float('inf')) > 2.0
        high_fitness = result.get('fitness', -float('inf')) >= 1.5

        return passes_tests or high_sharpe or high_fitness

    def _evaluate_single_program(
        self, program: Program
    ) -> tuple[str, AlphaPerf | None]:
        """
        Evaluate the fitness of a single program, handling retries and session recreation.

        Returns:
        -------
        tuple[str, AlphaPerf | None]: (program_str, performance or None if skipped)
        """
        program_str = str(program)

        # Skip if already evaluated
        if program_str in self.evaluated_expressions:
            return program_str, None

        # Try to evaluate with reconnection attempts if needed
        result: AlphaPerf | None = None
        for attempt in range(MAX_RECONNECTION_ATTEMPTS):
            try:
                # Get current session safely
                with self._session_lock:
                    current_session = self.session
                    metric_func = self.metric

                # Check if session is valid
                if current_session is None:
                    self.logger.warning('Session is None, attempting to recreate.')
                    with self._session_lock:
                        if not self._recreate_session():
                            self.logger.error(
                                'Failed to recreate session, cannot evaluate.'
                            )
                            break
                        metric_func = evaluate_fitness(self.session, logger=self.logger)

                # Attempt to evaluate the program
                result = metric_func(program_str)

                # Handle authentication failure
                if result is None:
                    with self._session_lock:
                        try:
                            # Check session status
                            test_response = self.session.get(
                                'https://api.worldquantbrain.com/authentication',
                                timeout=10,
                            )
                            if test_response.status_code != 200:
                                self.logger.warning(
                                    f'Session check failed ({test_response.status_code}), recreating.'
                                )
                                if self._recreate_session():
                                    self.metric = evaluate_fitness(
                                        self.session, logger=self.logger
                                    )
                                    continue
                            else:
                                self.logger.log(
                                    'Session check OK, but metric returned None. Retrying.'
                                )
                        except requests.exceptions.RequestException as e:
                            self.logger.warning(
                                f'Session check error: {e}. Attempting recreation.'
                            )
                            if self._recreate_session():
                                self.metric = evaluate_fitness(
                                    self.session, logger=self.logger
                                )
                                continue
                    # If we reach here, either the session is valid but metric failed, or recreation failed
                    self.logger.error(
                        f'Metric returned None. Skipping program: {program_str[:50]}...'
                    )
                    break

                # Successful evaluation
                if result is not None:
                    break

            except requests.exceptions.Timeout:
                self.logger.warning(
                    f'Timeout: {program_str[:50]}... Attempt {attempt + 1}/{MAX_RECONNECTION_ATTEMPTS}'
                )
                if attempt < MAX_RECONNECTION_ATTEMPTS - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    result = cast(AlphaPerf, {'error': 'timeout'})

            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f'Request error: {program_str[:50]}... {e}. Attempt {attempt + 1}/{MAX_RECONNECTION_ATTEMPTS}'
                )
                if attempt < MAX_RECONNECTION_ATTEMPTS - 1:
                    time.sleep(2**attempt)
                else:
                    result = cast(AlphaPerf, {'error': f'request error: {e}'})

        # Always mark as evaluated to prevent retry
        self.evaluated_expressions.add(program_str)

        # Process successful evaluation result
        if result is not None and 'error' not in result:
            self.fitness_evaluations += 1

            # Calculate fitness with parsimony pressure
            program.raw_fitness = result.get('fitness', 0)
            penalty = self.parsimony_coefficient * program.length()
            program.fitness = program.raw_fitness - penalty
            result['final_fitness'] = program.fitness

            self.logger.log(
                f'Program fitness: {program_str[:50]}... = {program.fitness:.4f}'
            )

            # Update Hall of Fame if result meets threshold
            if self._meets_hof_threshold(result):
                fitness_for_hof: float = (
                    result.get('fitness') or result.get('sharpe') or -float('inf')
                )
                entry = (fitness_for_hof, program_str, result)

                if len(self.hall_of_fame) < self.hof_size:
                    heapq.heappush(self.hall_of_fame, entry)
                else:
                    heapq.heappushpop(self.hall_of_fame, entry)

                if self.init_population_save_path is not None:
                    self._save_to_initial_population(
                        self.init_population_save_path, program.program
                    )
        else:
            # Handle evaluation failure
            program.raw_fitness = float('-inf')
            program.fitness = float('-inf')
            self.logger.warning(f'Evaluation failed for: {program_str[:50]}...')

        return program_str, result

    def _save_to_initial_population(
        self, path: str | os.PathLike[str], rpn: list[Operator | Terminal]
    ) -> None:
        try:
            population_path = os.fspath(path)
            existing_programs: list[list[Operator | Terminal]] = []

            if os.path.exists(population_path):
                shutil.move(population_path, f'{path}.bak')
                self.logger.warning(
                    f'{population_path} existed, move to {population_path}.bak'
                )

            with open(population_path, 'wb') as f:
                pickle.dump(existing_programs, f)

        except Exception as e:
            self.logger.error(f'Error saving to initial population file: {e}')

    def _evaluate_fitness(self, program) -> tuple[float, float]:
        """Deprecated: Use parallel_evaluate_fitness."""
        warnings.warn(
            '_evaluate_fitness is deprecated. Use parallel_evaluate_fitness instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        # Call _evaluate_single_program for consistency
        _, result = self._evaluate_single_program(program)
        return program.raw_fitness, program.fitness

    def parallel_evaluate_fitness(
        self, programs_to_evaluate: list[Program], n_parallel: int | None = None
    ):
        """
        Evaluate fitness for a list of programs in parallel.
        Manages evaluated_expressions and HOF updates.
        """
        if n_parallel is None:
            n_parallel = self.n_parallel

        # Filter out already evaluated programs
        to_evaluate = []
        for program in programs_to_evaluate:
            program_str = str(program)
            if program_str not in self.evaluated_expressions:
                to_evaluate.append(program)

        if not to_evaluate:
            return

        self.logger.log(
            f'Evaluating {len(to_evaluate)} programs (pool size {n_parallel})...'
        )

        start_time = time.time()

        # Run evaluations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = {
                executor.submit(self._evaluate_single_program, p): p
                for p in to_evaluate
            }

            for future in concurrent.futures.as_completed(futures):
                program = futures[future]
                try:
                    _, _ = (
                        future.result()
                    )  # We don't need to store results here as programs are updated in-place
                except Exception as exc:
                    program_str = str(program)
                    self.logger.error(
                        f'Program {program_str[:50]}... generated an exception: {exc}'
                    )
                    program.raw_fitness = float('-inf')
                    program.fitness = float('-inf')
                    self.evaluated_expressions.add(program_str)

        # Ensure all programs have fitness values
        for program in programs_to_evaluate:
            if program.fitness is None:
                program.raw_fitness = float('-inf')
                program.fitness = float('-inf')

        duration = time.time() - start_time
        self.logger.log(f'Batch evaluation completed in {duration:.2f}s')
        if self.hall_of_fame and len(self.hall_of_fame) > 0:
            best_hof = max(f[0] for f in self.hall_of_fame)
            self.logger.log(
                f'HOF: {len(self.hall_of_fame)} entries, best={best_hof:.4f}'
            )

    def _tournament_selection(self) -> Program:
        """Select a program using tournament selection."""
        indices = self.random_state.randint(
            0, len(self.population), self.tournament_size
        )
        tournament: list[Program] = [self.population[i] for i in indices]

        # Return the best individual in the tournament based on pre-calculated fitness
        # Handle potential None fitness values gracefully if evaluation failed.
        return max(
            tournament,
            key=lambda program: (
                program.fitness if program.fitness is not None else float('-inf')
            ),
        )

    def _update_best(self) -> None:
        """Update the best program found so far based on the main population."""
        # Find the best program in the current population
        current_best = max(
            self.population,
            key=lambda program: (
                program.fitness if program.fitness is not None else float('-inf')
            ),
        )

        # Handle case where fitness might be None
        if current_best.fitness is None:
            return

        # Update best program if the current best is better
        if current_best.fitness > self.best_fitness:
            self.best_program = current_best
            self.best_fitness = current_best.fitness

    def evolve(self, verbose=True, log_interval=1):
        """Evolve the population over generations."""
        if self.start_time is None:
            self.start_time = time.time()

        if not self.population:
            self._initialize_population()

        self.logger.log(f'Initial evaluation of {len(self.population)} programs...')

        needs_evaluation = [p for p in self.population if p.fitness is None]
        self.parallel_evaluate_fitness(needs_evaluation)
        self._update_best()

        if verbose:
            self.logger.log(
                f'Generation {self.generation}: Best Fitness={self.best_fitness:.4f}'
            )

        for gen in range(1, self.generations + 1):
            self.generation = gen
            start_gen_time = time.time()

            next_population: list[Program] = []
            next_gen_strings: set[str] = set()

            # Add elitism: Keep the best program from the previous generation
            if self.best_program is not None:
                best_str = str(self.best_program)
                next_population.append(self.best_program)
                next_gen_strings.add(best_str)
                if gen % log_interval == 0:
                    self.logger.log(
                        f'Elite: {best_str[:50]}... (Fitness: {self.best_fitness:.4f})'
                    )

            # Generate new population through crossover and mutation
            while len(next_population) < self.population_size:
                operation = self.random_state.choice(
                    ['crossover', 'mutation'], p=[self.p_crossover, self.p_mutation]
                )

                if operation == 'crossover':
                    parent1 = self._tournament_selection()
                    parent2 = self._tournament_selection()

                    # Ensure parents are different
                    attempts = 0
                    while parent1 is parent2 and attempts < 10:
                        parent2 = self._tournament_selection()
                        attempts += 1

                    # Get offspring programs
                    offspring1_program, _, _ = parent1.crossover(
                        parent2.program, self.random_state
                    )
                    offspring2_program, _, _ = parent2.crossover(
                        parent1.program, self.random_state
                    )

                    # Create new Program instances
                    offspring1 = Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring1_program,
                    )

                    offspring2 = Program(
                        max_depth=self.max_depth,
                        max_operators=self.max_operators,
                        random_state=self.random_state,
                        metric=self.metric,
                        parimony_coefficient=self.parsimony_coefficient,
                        program=offspring2_program,
                    )

                    offspring = [offspring1, offspring2]
                else:  # Mutation
                    parent = self._tournament_selection()
                    mutation_op = self.random_state.uniform()

                    if mutation_op < self.p_subtree_mutation:
                        # subtree_mutation returns (program, removed, donor_removed)
                        mutation_result = parent.subtree_mutation(self.random_state)
                        offspring_program = mutation_result[0]
                    elif mutation_op < self.p_subtree_mutation + self.p_hoist_mutation:
                        # hoist_mutation returns (program, removed)
                        offspring_program, _ = parent.hoist_mutation(self.random_state)
                    else:
                        # point_mutation returns (program, mutated_indices)
                        offspring_program, _ = parent.point_mutation(self.random_state)

                    offspring = [
                        Program(
                            max_depth=self.max_depth,
                            max_operators=self.max_operators,
                            random_state=self.random_state,
                            metric=self.metric,
                            parimony_coefficient=self.parsimony_coefficient,
                            program=offspring_program,
                        )
                    ]

                # Add unique offspring to next generation
                for child in offspring:
                    child_str = str(child)
                    if child_str not in next_gen_strings:
                        next_population.append(child)
                        next_gen_strings.add(child_str)
                        if len(next_population) >= self.population_size:
                            break  # Exit loop when population is full

                # Exit outer loop if population is full
                if len(next_population) >= self.population_size:
                    break

            # Evaluate the new generation
            self.logger.log(f'\n--- Generation {gen} ---')
            self.parallel_evaluate_fitness(next_population)

            # Update population and best program
            self.population = next_population
            self._update_best()

            # Store history with consistent metrics
            avg_fitness = np.mean(
                [
                    p.fitness
                    for p in self.population
                    if p.fitness is not None and p.fitness > -np.inf
                ]
            )
            best_hof_fitness = max(
                (f[0] for f in self.hall_of_fame), default=-float('inf')
            )

            self.history.append(
                {
                    'generation': gen,
                    'best_fitness': self.best_fitness,
                    'avg_fitness': avg_fitness,
                    'population_size': len(self.population),
                    'hof_size': len(self.hall_of_fame),
                    'best_hof_fitness': best_hof_fitness,
                }
            )

            # Log progress if necessary
            end_gen_time = time.time()
            gen_duration = end_gen_time - start_gen_time

            if verbose and (gen % log_interval == 0 or gen == self.generations):
                elapsed_time = timedelta(seconds=int(end_gen_time - self.start_time))
                self.logger.log(
                    f'Gen {gen:>{len(str(self.generations))}}: '
                    f'Best={self.best_fitness:.4f}, '
                    f'Avg={avg_fitness:.4f}, '
                    f'HOF Best={best_hof_fitness:.4f} (Size:{len(self.hall_of_fame)}), '
                    f'Evals={self.fitness_evaluations}, '
                    f'Time={gen_duration:.2f}s, '
                    f'Total={elapsed_time}'
                )

        return self.best_program

    def get_best_individual(self):
        """Returns the best individual found during the evolution (from HOF)."""
        if not self.hall_of_fame:
            return self.best_program  # Fall back to population best if HOF is empty

        # HOF stores (fitness, program_str, AlphaPerf), sorted smallest first by heapq
        best_entry = max(self.hall_of_fame, key=lambda item: item[0])
        fitness, program_str, result_dict = best_entry

        # Return dictionary with program information
        return {
            'program_string': program_str,
            'fitness': fitness,
            'result_details': result_dict,
        }

    def get_hall_of_fame(self):
        """Returns the entire Hall of Fame, sorted by fitness descending."""
        # Sort by fitness (descending) before returning
        return sorted(self.hall_of_fame, key=lambda item: item[0], reverse=True)

    def get_fitness_history(self):
        """Return the history of best fitness values."""
        return [stats['best_fitness'] for stats in self.history]

    def get_all_history(self):
        """Return all tracked statistics."""
        return self.history
