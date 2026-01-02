import time
from src.utils.python import run_python_code
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))


def test_security():
    print("Running security tests...")

    # Test 1: Valid code
    print("\nTest 1: Valid code")
    code = "print('Hello World')"
    result = run_python_code(code)
    print(f"Result: {result}")
    assert "Hello World" in result

    # Test 2: Forbidden import (os)
    print("\nTest 2: Forbidden import (os)")
    code = "import os\nprint(os.name)"
    result = run_python_code(code)
    print(f"Result: {result}")
    assert "Security Violation" in result

    # Test 3: Forbidden import (subprocess)
    print("\nTest 3: Forbidden import (subprocess)")
    code = "import subprocess\nsubprocess.run('ls')"
    result = run_python_code(code)
    print(f"Result: {result}")
    assert "Security Violation" in result

    # Test 4: Forbidden function (open)
    print("\nTest 4: Forbidden function (open)")
    code = "f = open('test.txt', 'w')"
    result = run_python_code(code)
    print(f"Result: {result}")
    assert "Security Violation" in result

    # Test 5: Timeout
    print("\nTest 5: Timeout (using 2s timeout)")
    code = "import time\nwhile True:\n    time.sleep(1)"
    start_time = time.time()
    result = run_python_code(code, timeout=2)
    end_time = time.time()
    print(f"Result: {result}")
    print(f"Time taken: {end_time - start_time:.2f}s")
    assert "Process timed out" in result
    assert end_time - start_time >= 2

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_security()
