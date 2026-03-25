import unittest
import numpy as np
from sympy import Integer as Int
from src.function import Operator, ADD, SUB, MUL, DIV, OPEN, HIGH, LOW, VOLUME
from src.program import Program


class TestProgramUnit(unittest.TestCase):
    def setUp(self):
        # Set up a fixed random state for reproducibility
        self.random_state = np.random.RandomState(42)
        # Mock arities and metric for test purposes
        self.arities = {2: [ADD, SUB, MUL, DIV]}
        self.metric = lambda x: 0  # Dummy metric

    def test_single_terminal_unit(self):
        """Test unit property with a single terminal node."""
        # Create a program with just one terminal
        program = [OPEN]  # OPEN has unit 3
        prog = Program(
            arities=self.arities,
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=program,
        )

        # Check that the unit matches the terminal's unit
        self.assertEqual(prog.unit[0], Int(3))
        self.assertEqual(len(prog.unit), 1)

    def test_simple_operator_unit(self):
        """Test unit property with a simple operator and terminals."""
        # Create a program: OPEN HIGH ADD
        # This should be equivalent to ADD(OPEN, HIGH) both with unit 3
        program = [OPEN, HIGH, ADD]
        prog = Program(
            arities=self.arities,
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=program,
        )

        # Check units for each node
        self.assertEqual(prog.unit[0], Int(3))  # OPEN has unit 3
        self.assertEqual(prog.unit[1], Int(3))  # HIGH has unit 3
        self.assertEqual(prog.unit[2], Int(3))  # ADD(3,3) has unit 3
        self.assertEqual(len(prog.unit), 3)

    def test_complex_program_unit(self):
        """Test unit property with a more complex program structure."""
        # Create a more complex program: OPEN HIGH ADD VOLUME MUL
        # This should be equivalent to MUL(ADD(OPEN, HIGH), VOLUME)
        # ADD(OPEN, HIGH) has unit 3, VOLUME has unit 2, MUL should multiply the units
        program = [OPEN, HIGH, ADD, VOLUME, MUL]
        prog = Program(
            arities=self.arities,
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=program,
        )

        # Check if the unit calculates correctly for MUL (should be 3*2 = 6)
        units = prog.unit
        self.assertEqual(units[0], Int(3))  # OPEN has unit 3
        self.assertEqual(units[1], Int(3))  # HIGH has unit 3
        self.assertEqual(units[2], Int(3))  # ADD(3,3) has unit 3
        self.assertEqual(units[3], Int(2))  # VOLUME has unit 2
        self.assertEqual(units[4], Int(6))  # MUL(3,2) has unit 6 (3*2)

    def test_unit_caching(self):
        """Test that unit values are cached and not recomputed unnecessarily."""
        program = [OPEN, HIGH, ADD]
        prog = Program(
            arities=self.arities,
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=program,
        )

        # Access unit first time to compute
        units_first = prog.unit

        # Modify internal _unit directly to test caching
        prog._unit[0] = Int(999)  # Change the first unit to a dummy value

        # Access unit again - should use cached value
        units_second = prog.unit

        # Check that it's using the modified value, not recomputing
        self.assertEqual(units_second[0], Int(999))

        # Now test if we make a new program object with the same program
        # First save the current program
        old_program = prog.program

        # Create a new different program
        new_program = [LOW, HIGH, ADD]

        # Explicitly replace the program
        prog.program = new_program

        # Now the _unit should be recomputed completely
        prog._unit = None  # Force reset of the cache for this test
        units_third = prog.unit

        # Should have recomputed correct values
        self.assertEqual(units_third[0], Int(3))  # LOW unit
        self.assertEqual(units_third[1], Int(3))  # HIGH unit
        self.assertEqual(units_third[2], Int(3))  # ADD unit

    def test_incompatible_units(self):
        """Test that incompatible units raise appropriate errors."""

        # Create custom operators for testing incompatible units
        # This operator should specifically raise an error for testing
        def failing_unit_rule(x, y):
            if x != y:
                raise ValueError('Units must be the same')
            return x

        TEST_OP = Operator(
            'test_op', 2, lambda x, y: f'test({x},{y})', failing_unit_rule
        )

        # Create a program with incompatible units: OPEN VOLUME TEST_OP
        program = [OPEN, VOLUME, TEST_OP]
        prog = Program(
            arities={2: [TEST_OP]},
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=program,
            skip_validation=True,  # Skip validation at init time
        )

        # Verify error is raised with correct message
        with self.assertRaises(ValueError) as context:
            units = prog.unit
        self.assertIn('Units must be the same', str(context.exception))
        self.assertIn(
            'Unit rule failed for operator test_op with units [3, 2]',
            str(context.exception),
        )

    def test_validate_program_with_units(self):
        """Test that validate_program correctly checks unit compatibility."""
        # 1. Create a valid program (same units for ADD)
        valid_program = [OPEN, HIGH, ADD]
        valid_prog = Program(
            arities=self.arities,
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=valid_program,
        )

        # Should validate successfully
        self.assertTrue(valid_prog.validate_program())

        # 2. Create a program with incompatible units for ADD
        def failing_unit_rule(x, y):
            if x != y:
                raise ValueError('Units must be the same')
            return x

        STRICT_ADD = Operator(
            'strict_add', 2, lambda x, y: f'add({x},{y})', failing_unit_rule
        )

        invalid_program = [
            OPEN,
            VOLUME,
            STRICT_ADD,
        ]  # OPEN has unit 3, VOLUME has unit 2
        invalid_prog = Program(
            arities={2: [STRICT_ADD]},
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=invalid_program,
            skip_validation=True,  # Skip validation at init time
        )

        # Should fail validation
        self.assertFalse(invalid_prog.validate_program())

        # 3. Test with MUL which should accept different units
        mul_program = [
            OPEN,
            VOLUME,
            MUL,
        ]  # OPEN has unit 3, VOLUME has unit 2, MUL should work
        mul_prog = Program(
            arities=self.arities,
            max_depth=3,
            max_operators=5,
            metric=self.metric,
            random_state=self.random_state,
            program=mul_program,
        )

        # Should validate successfully
        self.assertTrue(mul_prog.validate_program())


if __name__ == '__main__':
    unittest.main()
