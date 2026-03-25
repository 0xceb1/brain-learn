import unittest
import numpy as np
import time
from src.genetic import GPLearnSimulator


class RealisticMetric:
    """A more realistic metric function for simulating a financial alpha factor.

    Simulates backtesting by evaluating expressions based on their financial
    properties and robustness across different market regimes.
    """

    def __init__(self):
        # Initialize with set of preferred characteristics
        self.preferred_terms = {
            'volume': 0.4,  # Volume factor
            'adv20': 0.2,  # Average daily volume
            'returns': 0.3,  # Return term
            'high': 0.2,  # High price
            'low': 0.2,  # Low price
            'close': 0.1,  # Close price
            'open': 0.1,  # Open price
        }

        # Preferred operator combinations
        self.operator_bonuses = {
            'rank': 0.3,
            'divide': 0.25,
            'ts_corr': 0.35,
            'ts_zscore': 0.4,
        }

        # Noise level for fitness (simulates market noise)
        self.noise_level = 0.05

        # Track expressions we've seen to simulate overfitting penalty
        self.seen_expressions = set()

        # Track calls for performance metrics
        self.call_count = 0
        self.execution_time = 0

    def __call__(self, expr):
        """Calculate a simulated Sharpe ratio for the given expression."""
        start_time = time.time()
        self.call_count += 1

        try:
            # Base score
            score = 0.1

            # Add scores for preferred terms
            for term, value in self.preferred_terms.items():
                if term in expr:
                    score += value

            # Add scores for preferred operators
            for op, value in self.operator_bonuses.items():
                if op in expr:
                    score += value

            # Reward expressions with multiple operations (moderate complexity)
            operations = expr.count('(')
            if operations > 1:
                score += min(operations * 0.1, 0.5)  # Cap at 0.5

            # Penalty for overly complex expressions
            if len(expr) > 150:
                score *= 0.7

            # Penalty for previously seen expressions (to simulate overfitting penalty)
            if expr in self.seen_expressions:
                score *= 0.8
            self.seen_expressions.add(expr)

            # Add noise to simulate market variance
            score += np.random.normal(0, self.noise_level)

            # Clamp to reasonable range
            score = max(0, min(score, 2.0))

            self.execution_time += time.time() - start_time
            return {'sharpe': score}

        except Exception as e:
            self.execution_time += time.time() - start_time
            print(f'Error evaluating expression: {e}')
            return {'sharpe': None}

    def get_stats(self):
        """Return statistics about metric performance."""
        return {
            'calls': self.call_count,
            'avg_time': self.execution_time / max(1, self.call_count),
            'total_time': self.execution_time,
        }


class TestIntegration(unittest.TestCase):
    def test_optimization_run(self):
        """Test a complete optimization run with realistic settings."""
        # Create the metric
        metric = RealisticMetric()

        # Create the simulator
        gp = GPLearnSimulator(
            population_size=20,
            generations=5,
            tournament_size=3,
            p_crossover=0.7,
            p_mutation=0.3,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.15,
            max_depth=3,
            max_operators=5,
            random_state=42,
            parsimony_coefficient=0.1,
        )
        gp.metric = metric

        # Run the evolution
        print('\nRunning optimization with realistic settings...')
        start_time = time.time()
        best_program = gp.evolve(verbose=True, log_interval=1)
        total_time = time.time() - start_time

        # Check the results
        print(f'\nBest program: {best_program}')
        print(f'Best fitness: {gp.best_fitness:.4f}')
        print(f'Best program length: {len(best_program.program)}')

        # Check metric performance
        metric_stats = metric.get_stats()
        print('\nMetric stats:')
        print(f'Total calls: {metric_stats["calls"]}')
        print(f'Average evaluation time: {metric_stats["avg_time"]:.6f} seconds')
        print(f'Total evaluation time: {metric_stats["total_time"]:.2f} seconds')
        print(f'Total optimization time: {total_time:.2f} seconds')
        print(
            f'Evaluation time percentage: {(metric_stats["total_time"] / total_time) * 100:.1f}%'
        )

        # Verify that the optimization produced good results
        self.assertIsNotNone(best_program)
        self.assertGreater(gp.best_fitness, 0.5)  # Should find a decent solution

        # Fitness shouldn't decrease over generations (elitism should ensure this)
        # But might not increase due to stochastic nature, so we check for non-decrease
        fitness_history = gp.get_fitness_history()
        for i in range(1, len(fitness_history)):
            self.assertGreaterEqual(
                fitness_history[i],
                fitness_history[i - 1],
                f'Fitness decreased from {fitness_history[i - 1]} to {fitness_history[i]} at generation {i}',
            )

        # Check for some basic indicators of a reasonable solution
        program_str = str(best_program)
        has_valid_operators = any(
            op in program_str
            for op in ['ts_corr', 'divide', 'rank', 'subtract', 'multiply']
        )
        self.assertTrue(
            has_valid_operators, 'Best program should contain some valid operators'
        )


if __name__ == '__main__':
    unittest.main()
