from pathlib import Path
import subprocess
import sys


SCRIPTS = [
    Path("processing") / "1process-data.py",
    Path("processing") / "2find-missing.py",
    Path("processing") / "3remove-unreliable-columns.py",
    Path("processing") / "4find-missing-from-filtered.py",
    Path("processing") / "5fill-missing-values.py",
    Path("processing") / "6generate-shap.py",
]


def main() -> None:
    for script in SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f"""Missing script: {script}""")

        cmd = [sys.executable, str(script)]
        print(f"""\nRunning: {' '.join(cmd)}""")
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            raise SystemExit(f"""Script failed ({script}) with exit code {completed.returncode}""")


if __name__ == "__main__":
    main()
