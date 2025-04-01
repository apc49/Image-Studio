import subprocess
import os

def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running {command}: {str(e)}"

log_path = "debug_report.txt"
with open(log_path, "w", encoding="utf-8") as log:
    log.write("==== AWDigitalworld ImageAnalyzer Code Validation ====

")
    
    log.write("STEP 1: Running black (autoformatter)...\n")
    log.write(run_command("black main.py --check"))  # Only check, don’t format automatically
    log.write("\n\n")

    log.write("STEP 2: Running flake8 (style + errors)...\n")
    log.write(run_command("flake8 main.py"))
    log.write("\n\n")

    log.write("STEP 3: Running pylint (deep analysis)...\n")
    log.write(run_command("pylint main.py --disable=R,C"))  # Skip style warnings, focus on issues
    log.write("\n\n")

print(f"✅ Validation complete. Report saved to {log_path}")
