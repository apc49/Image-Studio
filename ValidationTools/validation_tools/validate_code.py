import os
import subprocess
import py_compile

print("==== AWDigitalworld ImageAnalyzer Code Validation ====")

# Check requirements.txt
print("\nChecking for missing packages...")
if os.path.exists("requirements.txt"):
    result = subprocess.run(["pip", "check"], capture_output=True, text=True)
    print(result.stdout or "✓ All packages satisfied.")

# Syntax validation
print("\nSTEP 2: Checking syntax errors in main.py...")
try:
    py_compile.compile("main.py", doraise=True)
    print("✓ No syntax errors found in main.py.")
except py_compile.PyCompileError as e:
    print("✗ Syntax error detected in main.py:", e)

# Dry run
print("\nSTEP 3: Dry-run execution of main.py for hidden runtime errors...")
try:
    subprocess.run(["python", "-m", "trace", "--trace", "main.py"], check=True)
except subprocess.CalledProcessError:
    print("✗ Runtime errors found during dry-run.")
