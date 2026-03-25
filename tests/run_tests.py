"""
Test runner script for brain-learn
Run this script to execute all tests in the tests directory
"""

import importlib
import os
import sys


def run_all_tests():
    """Find and run all test functions in the tests directory"""
    test_files = []
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all Python files that start with "test_"
    for filename in os.listdir(test_dir):
        if filename.startswith('test_') and filename.endswith('.py'):
            test_files.append(filename[:-3])  # Remove .py extension

    print(f'Found {len(test_files)} test files: {", ".join(test_files)}')

    # Track test results
    all_passed = True
    passed_count = 0
    failed_count = 0

    # Run each test file
    for test_module in test_files:
        print(f'\n{"=" * 50}')
        print(f'Running tests from {test_module}...')

        # Import the module
        module = importlib.import_module(f'tests.{test_module}')

        # Find all test functions (those starting with "test_")
        test_functions = [
            name
            for name in dir(module)
            if name.startswith('test_') and callable(getattr(module, name))
        ]

        # Run each test function
        for test_func_name in test_functions:
            test_func = getattr(module, test_func_name)
            print(f'\n- Running {test_func_name}:')
            try:
                result = test_func()
                if result is None or result is True:
                    print(f'✅ {test_func_name} passed!')
                    passed_count += 1
                else:
                    print(f'❌ {test_func_name} failed!')
                    failed_count += 1
                    all_passed = False
            except Exception as e:
                print(f'❌ {test_func_name} raised an exception: {str(e)}')
                failed_count += 1
                all_passed = False

    # Print summary
    print(f'\n{"=" * 50}')
    print(f'Test Summary: {passed_count} passed, {failed_count} failed')

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    # Return exit code based on test results (useful for CI/CD)
    sys.exit(0 if success else 1)
