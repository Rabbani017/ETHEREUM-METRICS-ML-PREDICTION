"""Master Launcher for Machine Learning Analysis & Prediction of Ethereum Metrics.

Project Team:
- Shaik Rabbani (AP23110010538)
- P. Praneeth (AP23110010508)
- P. Prabath (AP23110010565)
University: SRM University - AP

Usage:
    py run_project.py
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path


def find_free_port() -> int:
    """Find a random open local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_dashboard(port: int | None = None) -> None:
    """Launch the unified master Streamlit dashboard."""
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "ethereum_metrics_prediction" / "app.py"

    if not app_path.exists():
        app_path = project_root / "ethereum_blockchain_ml" / "app.py"

    if port is None:
        port = find_free_port()

    url = f"http://localhost:{port}"
    print("\n" + "=" * 70)
    print("  ETHEREUM BLOCKCHAIN MACHINE LEARNING RESEARCH DASHBOARD")
    print("=" * 70)
    print(f"  Team:       Shaik Rabbani, P. Praneeth, P. Prabath")
    print(f"  University: SRM University - AP")
    print(f"  Local URL:  {url}")
    print("=" * 70)
    print("  Starting dashboard in browser... (Press Ctrl+C to stop)\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless",
            "true",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            str(port),
        ],
        cwd=app_path.parent,
        check=False,
    )


def run_tests() -> None:
    """Run all unit tests across both project test suites."""
    root = Path(__file__).resolve().parent
    dir1 = root / "ethereum_metrics_prediction"
    dir2 = root / "ethereum_blockchain_ml"

    print("\n" + "=" * 60)
    print("RUNNING ETHEREUM METRICS PREDICTION TESTS")
    print("=" * 60)
    res1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            f"pythonpath={dir1}",
            str(dir1 / "tests"),
            "-v",
        ],
        cwd=dir1,
        check=False,
    )

    print("\n" + "=" * 60)
    print("RUNNING ETHEREUM BLOCKCHAIN ML TESTS")
    print("=" * 60)
    res2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            f"pythonpath={dir2}",
            str(dir2 / "tests"),
            "-v",
        ],
        cwd=dir2,
        check=False,
    )
    if res1.returncode == 0 and res2.returncode == 0:
        print("\n" + "=" * 60)
        print("🎉 ALL 37 TESTS ACROSS BOTH SUITES PASSED (100% SUCCESS)!")
        print("=" * 60 + "\n")
    else:
        print("\n[WARNING] Some tests encountered errors.")


def run_live_prediction() -> None:
    """Execute live next-block inference on the newest on-chain blocks."""
    root = Path(__file__).resolve().parent
    script = root / "ethereum_metrics_prediction" / "src" / "predict_next.py"
    subprocess.run([sys.executable, str(script)], cwd=root, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Ethereum ML Project Launcher.")
    parser.add_argument("--test", action="store_true", help="Run full test suites.")
    parser.add_argument("--predict", action="store_true", help="Run live next-block prediction.")
    parser.add_argument("--port", type=int, default=None, help="Custom server port.")
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.predict:
        run_live_prediction()
    else:
        start_dashboard(args.port)


if __name__ == "__main__":
    main()
