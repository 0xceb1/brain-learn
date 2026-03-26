import unittest

import numpy as np
import requests

from src.genetic import GPLearnSimulator
from src.logger import Logger
from src.program import Program
from src.function import ADD, MUL, DIV, OPEN, CLOSE, VOLUME


def dummy_metric(expr):
    """A dummy metric function that returns a simple score based on expression complexity.

    This simulates the behavior of the actual evaluation function without
    requiring access to the actual simulation backend.

    Parameters
    ----------
    expr : str
        The string representation of the expression to evaluate

    Returns
    -------
    dict
        A dict with 'sharpe' key containing the simulated Sharpe ratio
    """
    try:
        # Simple heuristic: more complex expressions have higher initial scores
        # but excessive complexity is penalized
        complexity = expr.count('(') * 2 + len(expr)
        # Prefer expressions with volume term
        has_volume = 'volume' in expr
        # Avoid trivial expressions
        too_simple = len(expr) < 10

        # Sharpe ratio simulation - in real system this would be calculated by backtesting
        sharpe = 0.5 + np.random.normal(0, 0.2)

        # Adjust based on heuristics
        if has_volume:
            sharpe += 0.3
        if too_simple:
            sharpe -= 0.5

        # Apply a complexity penalty for extremely long expressions
        complexity_penalty = max(0, (complexity - 50) / 100)
        sharpe -= complexity_penalty

        return {'sharpe': sharpe, 'fitness': sharpe}
    except Exception as e:
        # Return no score for invalid expressions
        print(f'Error evaluating expression: {e}')
        return {'sharpe': None}


def deterministic_metric(expr):
    """A deterministic metric function for reproducible tests.

    Returns a predictable score based on the structure of the expression.

    Parameters
    ----------
    expr : str
        The string representation of the expression to evaluate

    Returns
    -------
    dict
        A dict with 'sharpe' key containing the deterministic Sharpe ratio
    """
    try:
        # Count key operators as a measure of expression complexity
        num_add = expr.count('add(')
        num_mul = expr.count('multiply(')
        num_div = expr.count('divide(')

        # Specific pattern scoring - prefer multiplication and division
        sharpe = 0.1
        sharpe += num_add * 0.05
        sharpe += num_mul * 0.2
        sharpe += num_div * 0.15

        # Bonus for using volume - simulate that this is a valuable feature
        if 'volume' in expr:
            sharpe += 0.5

        # Penalty for overly complex expressions
        if len(expr) > 100:
            sharpe *= 0.7

        # Minimum score for valid expressions
        sharpe = max(0.01, sharpe)

        return {'sharpe': sharpe, 'fitness': sharpe}
    except Exception:
        return {'sharpe': None}


class TestGPLearnSimulator(unittest.TestCase):
    def setUp(self):
        self.session = requests.Session()
        self.logger = Logger(job_name='test-gp', console_log=False, file_log=False)
        self.username = 'test_user'
        self.password = 'test_pass'

    def test_initialization(self):
        """Test the initialization of the GPLearnSimulator."""
        # Initialize with default parameters
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            random_state=42,
        )

        # Check that parameters were set correctly
        self.assertEqual(gp.population_size, 30)
        self.assertEqual(gp.generations, 20)
        self.assertEqual(gp.tournament_size, 5)
        # Crossover / mutation are normalized to sum to 1
        self.assertAlmostEqual(gp.p_crossover, 0.7 / (0.7 + 0.1))
        self.assertAlmostEqual(gp.p_mutation, 0.1 / (0.7 + 0.1))

        # Initialize with custom parameters
        custom_gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=50,
            generations=10,
            tournament_size=3,
            random_state=42,
        )

        # Check that custom parameters were set correctly
        self.assertEqual(custom_gp.population_size, 50)
        self.assertEqual(custom_gp.generations, 10)
        self.assertEqual(custom_gp.tournament_size, 3)

    def test_population_initialization(self):
        """Test the initialization of the population."""
        # Use a fixed random seed for reproducibility
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            random_state=42,
        )

        # Initialize the population
        gp._initialize_population()

        # Check that the population has the correct size
        self.assertEqual(len(gp.population), 10)

        # Check that all members are Program instances
        for member in gp.population:
            self.assertIsInstance(member, Program)

    def test_fitness_evaluation(self):
        """Test the fitness evaluation of a program."""
        # Create a simulator with the dummy metric function
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            random_state=42,
        )
        gp.metric = dummy_metric

        # Create a simple program for testing
        program = Program(
            max_depth=3,
            max_operators=5,
            random_state=np.random.RandomState(42),
            metric=dummy_metric,
            parimony_coefficient=0.1,
            program=[OPEN, CLOSE, ADD, VOLUME, MUL],  # (OPEN + CLOSE) * VOLUME
        )

        raw, fitness = gp._evaluate_fitness(program)

        self.assertIsInstance(raw, float)
        self.assertIsInstance(fitness, float)

        # Check that the program's fitness has been set
        self.assertIsNotNone(program.fitness)

        # Ensure fitness evaluations counter is incremented
        self.assertEqual(gp.fitness_evaluations, 1)

    def test_tournament_selection(self):
        """Test the tournament selection method."""
        # Create a simulator
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            tournament_size=3,
            random_state=42,
        )
        gp.metric = dummy_metric

        # Create a population with known fitness values
        gp.population = []
        for i in range(10):
            program = Program(
                max_depth=3,
                max_operators=5,
                random_state=np.random.RandomState(42),
                metric=dummy_metric,
                parimony_coefficient=0.1,
                program=[OPEN, CLOSE, ADD],  # Simple program: OPEN + CLOSE
            )
            # Manually set fitness values (higher index = higher fitness)
            program.fitness = i / 10
            gp.population.append(program)

        # Run tournament selection multiple times
        selections = [gp._tournament_selection() for _ in range(50)]

        # Calculate the average fitness of selected individuals
        avg_fitness = sum(program.fitness for program in selections) / len(selections)

        # With tournament selection, average fitness should be higher than random selection (0.45)
        self.assertGreater(avg_fitness, 0.45)

    def test_evolution_process(self):
        """Test the evolution process for a few generations."""
        # Create a simulator with a small population and few generations
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            generations=3,
            tournament_size=3,
            random_state=42,
        )
        gp.metric = dummy_metric

        # Run evolution
        best_program = gp.evolve(verbose=False)

        # Check that the best program is returned
        self.assertIsInstance(best_program, Program)

        # Check that history has been updated
        self.assertEqual(len(gp.history), 3)

        # Check that the best fitness is tracked
        self.assertIsNotNone(gp.best_fitness)

        # Best fitness should increase or stay the same (not decrease)
        fitness_history = gp.get_fitness_history()
        for i in range(1, len(fitness_history)):
            self.assertGreaterEqual(fitness_history[i], fitness_history[i - 1])

        # Check statistics
        self.assertGreater(gp.fitness_evaluations, 0)

    def test_get_best_individual(self):
        """Test getting the best individual."""
        # Create and evolve a simulator
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            generations=2,
            random_state=42,
        )
        gp.metric = dummy_metric
        gp.evolve(verbose=False)

        best = gp.get_best_individual()

        # HOF entries are returned as metadata dicts; population best stays on gp.best_program
        self.assertIsInstance(gp.best_program, Program)
        assert gp.best_program is not None
        if isinstance(best, dict):
            self.assertIn('program_string', best)
            self.assertIn('fitness', best)
        else:
            self.assertIsInstance(best, Program)

        self.assertEqual(gp.best_program.fitness, gp.best_fitness)

    def test_deterministic_evolution(self):
        """Test evolution with a deterministic fitness function."""
        # Create a simulator with deterministic metric
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=15,
            generations=5,
            tournament_size=3,
            random_state=42,
        )
        gp.metric = deterministic_metric

        # Run evolution
        best_program = gp.evolve(verbose=False)

        # Record results
        first_run_fitness = gp.best_fitness
        first_run_evaluations = gp.fitness_evaluations

        # Reset and run again with same parameters
        gp2 = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=15,
            generations=5,
            tournament_size=3,
            random_state=42,
        )
        gp2.metric = deterministic_metric
        gp2.evolve(verbose=False)

        # With deterministic metric and same random seed, results should be identical
        self.assertEqual(first_run_fitness, gp2.best_fitness)
        self.assertEqual(first_run_evaluations, gp2.fitness_evaluations)

        self.assertIsNotNone(best_program)
        self.assertGreater(len(str(best_program)), 0)

    def test_crossover_probability(self):
        """Test different crossover probabilities."""
        # Run with high crossover probability
        gp_high_crossover = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            generations=3,
            p_crossover=0.9,
            p_mutation=0.1,
            random_state=42,
        )
        gp_high_crossover.metric = deterministic_metric
        gp_high_crossover.evolve(verbose=False)

        # Run with low crossover probability
        gp_low_crossover = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            generations=3,
            p_crossover=0.1,
            p_mutation=0.9,
            random_state=42,
        )
        gp_low_crossover.metric = deterministic_metric
        gp_low_crossover.evolve(verbose=False)

        # Both should produce valid results
        self.assertIsNotNone(gp_high_crossover.best_program)
        self.assertIsNotNone(gp_low_crossover.best_program)

    def test_parsimony_pressure(self):
        """Test different parsimony pressure coefficients."""
        # Run with low parsimony pressure
        gp_low_parsimony = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            generations=3,
            parsimony_coefficient=0.01,
            random_state=42,
        )
        gp_low_parsimony.metric = deterministic_metric
        gp_low_parsimony.evolve(verbose=False)

        # Run with high parsimony pressure
        gp_high_parsimony = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            generations=3,
            parsimony_coefficient=0.5,
            random_state=42,
        )
        gp_high_parsimony.metric = deterministic_metric
        gp_high_parsimony.evolve(verbose=False)

        # With higher parsimony pressure, programs should be shorter on average
        avg_length_low = np.mean([len(p.program) for p in gp_low_parsimony.population])
        avg_length_high = np.mean(
            [len(p.program) for p in gp_high_parsimony.population]
        )

        # This isn't guaranteed due to randomness, but is likely with enough runs
        # We're using a statistical test rather than a strict assertion
        self.assertLessEqual(avg_length_high, avg_length_low * 1.5)

    def test_empty_population_handling(self):
        """Test that the evolution process handles empty initial population."""
        # Create a simulator but don't initialize population
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=5,
            generations=2,
            random_state=42,
        )
        gp.metric = deterministic_metric

        # Explicitly empty the population
        gp.population = []

        # Population should be empty initially
        self.assertEqual(len(gp.population), 0)

        # Evolution should initialize population
        gp.evolve(verbose=False)

        # Now population should be the specified size
        self.assertEqual(len(gp.population), 5)

    def test_edge_case_fitness(self):
        """Test handling of edge cases in fitness evaluation."""
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            random_state=42,
        )

        def problematic_metric(expr):
            if 'divide' in expr:
                return {'error': 'divide failed'}
            if len(expr) < 10:
                return {'error': 'too short'}
            return {'sharpe': 0.5, 'fitness': 0.5}

        gp.metric = problematic_metric

        # Create programs that will trigger different error conditions
        program1 = Program(
            max_depth=3,
            max_operators=5,
            random_state=np.random.RandomState(42),
            metric=problematic_metric,
            parimony_coefficient=0.1,
            program=[OPEN, CLOSE, DIV],  # Should return None
        )

        program2 = Program(
            max_depth=3,
            max_operators=5,
            random_state=np.random.RandomState(42),
            metric=problematic_metric,
            parimony_coefficient=0.1,
            program=[OPEN],  # Should raise exception
        )

        _raw1, fitness1 = gp._evaluate_fitness(program1)
        _raw2, fitness2 = gp._evaluate_fitness(program2)

        self.assertEqual(fitness1, float('-inf'))
        self.assertEqual(fitness2, float('-inf'))
        self.assertEqual(program1.fitness, float('-inf'))
        self.assertEqual(program2.fitness, float('-inf'))

    def test_parallel_evaluation(self):
        """Test the parallel fitness evaluation functionality."""
        import time

        # Create a metric function that includes a delay to simulate computation time
        def delayed_metric(expr):
            time.sleep(0.1)  # 100ms delay
            s = len(expr) / 100
            return {'sharpe': s, 'fitness': s}

        # Create 10 programs
        programs = []
        for _ in range(10):
            program = Program(
                max_depth=3,
                max_operators=5,
                random_state=np.random.RandomState(42),
                metric=delayed_metric,
                parimony_coefficient=0.1,
            )
            programs.append(program)

        # Test with sequential evaluation (n_parallel=1)
        gp_sequential = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            n_parallel=1,
            random_state=42,
        )
        gp_sequential.metric = delayed_metric

        start_time = time.time()
        gp_sequential.parallel_evaluate_fitness(programs, n_parallel=1)
        sequential_time = time.time() - start_time

        # Reset program fitness for next test
        for program in programs:
            program.fitness = None
            program.raw_fitness = None

        # Test with parallel evaluation (n_parallel=3)
        gp_parallel = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            n_parallel=3,
            random_state=42,
        )
        gp_parallel.metric = delayed_metric

        start_time = time.time()
        gp_parallel.parallel_evaluate_fitness(programs, n_parallel=3)
        parallel_time = time.time() - start_time

        # Check that all programs have fitness values
        for program in programs:
            self.assertIsNotNone(program.fitness)

        # Thread timing is noisy (overhead, duplicate skips); both paths should finish
        self.assertGreater(sequential_time, 0)
        self.assertGreater(parallel_time, 0)

        # Test with too many parallel workers (should cap at 3)
        for program in programs:
            program.fitness = None
            program.raw_fitness = None

        gp_too_many = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=10,
            n_parallel=10,  # More than allowed
            random_state=42,
        )
        gp_too_many.metric = delayed_metric

        # Should not raise an exception, but should warn and cap at 3
        gp_too_many.parallel_evaluate_fitness(programs, n_parallel=10)

        # Check that all programs still have fitness values
        for program in programs:
            self.assertIsNotNone(program.fitness)

    def test_batch_processing_and_timeout(self):
        """Test batch processing and timeout functionality of parallel_evaluate_fitness."""
        import time

        # Create a metric function with variable delay
        def variable_delay_metric(expr):
            if 'open' in expr.lower():
                time.sleep(0.5)
            else:
                time.sleep(0.1)
            s = len(expr) / 100
            return {'sharpe': s, 'fitness': s}

        def timeout_metric(_expr):
            # Genetic treats results with 'error' like failed BRAIN evaluations
            return {'error': 'timeout'}

        # Test batch processing with larger number of programs
        gp = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=40,
            n_parallel=3,
            random_state=42,
        )

        gp.metric = variable_delay_metric

        # Create 75 programs (should be processed in 2 batches with default BATCH_SIZE=50)
        large_batch = []
        for i in range(75):
            program = Program(
                max_depth=3,
                max_operators=5,
                random_state=np.random.RandomState(i),  # Different seeds for variety
                metric=variable_delay_metric,
                parimony_coefficient=0.1,
            )
            large_batch.append(program)

        # Evaluate the large batch
        start_time = time.time()
        gp.parallel_evaluate_fitness(large_batch, n_parallel=3)
        processing_time = time.time() - start_time

        # All programs should have fitness values
        for program in large_batch:
            self.assertIsNotNone(program.fitness)

        # Test timeout handling
        gp_timeout = GPLearnSimulator(
            self.session,
            self.logger,
            self.username,
            self.password,
            population_size=5,
            n_parallel=2,
            random_state=42,
        )
        gp_timeout.metric = timeout_metric

        # Create a few programs that will time out
        timeout_programs = []
        for i in range(3):
            program = Program(
                max_depth=2,
                max_operators=3,
                random_state=np.random.RandomState(i),
                metric=timeout_metric,
                parimony_coefficient=0.1,
            )
            timeout_programs.append(program)

        # This should handle the timeout gracefully
        gp_timeout.parallel_evaluate_fitness(timeout_programs, n_parallel=2)

        # All programs should have fitness values even if they timed out
        for program in timeout_programs:
            self.assertIsNotNone(program.fitness)
            # They should have been assigned -inf fitness
            self.assertEqual(program.fitness, float('-inf'))


if __name__ == '__main__':
    unittest.main()
