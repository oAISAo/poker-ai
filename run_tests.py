import subprocess
import sys

# python run_tests.py

def run_tests():
    print("🔍 Running all tests in the 'test/' folder...\n")
    result = subprocess.run(
        ['pytest', 'test/', '-v', '--cov=.', '--cov-report=term-missing'],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. See output above.")
        sys.exit(result.returncode)

if __name__ == '__main__':
    run_tests()


