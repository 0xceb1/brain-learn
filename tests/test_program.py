from src.function import (
    Terminal,
    Operator,
    ADD,
    SUB,
    MUL,
    DIV,
    OPEN,
    CLOSE,
    HIGH,
    LOW,
    RANK,
    SQRT,
)
from src.program import Program


def program_to_readable(program):
    """Convert a program list to a human-readable string representation"""
    return ' → '.join([node.name for node in program])


def visualize_program(program):
    """Visualize a program structure as a tree without modifying the Program class.

    Args:
        program: List of Terminal and Operator objects in postfix notation

    Returns:
        str: A multi-line string showing the expression tree
    """
    if not program:
        return 'Empty Program'

    # Tree node class for visualization
    class TreeNode:
        def __init__(self, value, children=None):
            self.value = value
            self.children = children or []

    # Simulate the evaluation stack to build the tree
    stack = []

    # Build the tree from the postfix expression
    for node in program:
        if isinstance(node, Terminal):
            # For terminals, create a leaf node
            stack.append(TreeNode(node.name))
        elif isinstance(node, Operator):
            # For operators, pop operands and create a node with children
            if len(stack) < node.arity:
                return f'Invalid program: not enough operands for {node.name}'

            children = []
            for _ in range(node.arity):
                children.insert(0, stack.pop())

            # Create parent node with these children
            stack.append(TreeNode(node.name, children))

    # The stack should have exactly one node - the root
    if len(stack) != 1:
        return f'Invalid program structure: expected 1 root but got {len(stack)}'

    root = stack[0]

    # Helper function to print the tree recursively
    def print_tree(node, prefix='', is_last=True):
        result = []
        # Print the current node
        result.append(f'{prefix}{"└── " if is_last else "├── "}{node.value}')

        # Print children
        prefix += '    ' if is_last else '│   '
        for i, child in enumerate(node.children):
            is_last_child = i == len(node.children) - 1
            result.extend(print_tree(child, prefix, is_last_child))

        return result

    # Start printing from the root
    lines = [root.value]  # Root node
    prefix = ''
    for i, child in enumerate(root.children):
        is_last_child = i == len(root.children) - 1
        lines.extend(print_tree(child, prefix, is_last_child))

    return '\n'.join(lines)


def test_str_method():
    """Test the string representation of Program class"""
    # Create a simple program: (open + close) * (high - low)
    # This should generate: multiply(add(open,close),subtract(high,low))
    # In postfix notation: open close ADD high low SUB MUL
    program = [
        OPEN,  # First operand
        CLOSE,  # Second operand
        ADD,  # Add operation (open + close)
        HIGH,  # Third operand
        LOW,  # Fourth operand
        SUB,  # Subtract operation (high - low)
        MUL,  # Multiply operation ((open + close) * (high - low))
    ]

    # Create Program instance with the test program
    p = Program(
        max_depth=3,
        max_operators=5,
        metric=lambda x: 0,
        parimony_coefficient=0.1,
        random_state=None,
        program=program,
    )

    # Get the string representation
    result = str(p)
    expected = 'multiply(add(open,close),subtract(high,low))'

    # Print results
    print(f'Program (postfix): {program_to_readable(program)}')
    print(f'Result:   {result}')
    print(f'Expected: {expected}')
    print(f'Test passed: {result == expected}')
    print('\nExpression structure:')
    print(visualize_program(program))

    # Test a more complex program: add(divide(open,close),multiply(high,low))
    # In postfix notation: open close DIV high low MUL ADD
    complex_program = [
        OPEN,  # First operand
        CLOSE,  # Second operand
        DIV,  # Divide operation (open / close)
        HIGH,  # Third operand
        LOW,  # Fourth operand
        MUL,  # Multiply operation (high * low)
        ADD,  # Add operation ((open / close) + (high * low))
    ]

    p2 = Program(
        max_depth=3,
        max_operators=5,
        metric=lambda x: 0,
        parimony_coefficient=0.1,
        random_state=None,
        program=complex_program,
    )

    result2 = str(p2)
    expected2 = 'add(divide(open,close),multiply(high,low))'

    print('\nTest case 2:')
    print(f'Program (postfix): {program_to_readable(complex_program)}')
    print(f'Result:   {result2}')
    print(f'Expected: {expected2}')
    print(f'Test passed: {result2 == expected2}')
    print('\nExpression structure:')
    print(visualize_program(complex_program))

    # Test empty program
    empty_program = Program(
        max_depth=3,
        max_operators=5,
        metric=lambda x: 0,
        parimony_coefficient=0.1,
        random_state=None,
        program=[],
    )
    result3 = str(empty_program)
    expected3 = 'EmptyProgram'

    print('\nTest case 3 (Empty Program):')
    print('Program: []')
    print(f'Result:   {result3}')
    print(f'Expected: {expected3}')
    print(f'Test passed: {result3 == expected3}')

    # Test a more complex nested program: rank(sqrt(open*close))
    # In postfix notation: open close MUL SQRT RANK
    nested_program = [OPEN, CLOSE, MUL, SQRT, RANK]

    p4 = Program(
        max_depth=3,
        max_operators=5,
        metric=lambda x: 0,
        parimony_coefficient=0.1,
        random_state=None,
        program=nested_program,
    )

    result4 = str(p4)
    expected4 = 'rank(sqrt(multiply(open,close)))'

    print('\nTest case 4 (Nested operations):')
    print(f'Program (postfix): {program_to_readable(nested_program)}')
    print(f'Result:   {result4}')
    print(f'Expected: {expected4}')
    print(f'Test passed: {result4 == expected4}')
    print('\nExpression structure:')
    print(visualize_program(nested_program))

    return (
        result == expected
        and result2 == expected2
        and result3 == expected3
        and result4 == expected4
    )


if __name__ == '__main__':
    test_str_method()
