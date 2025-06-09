import os
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
REQUIRED_ENV = ["BINANCE_API_KEY", "BINANCE_SECRET_KEY", "DATABASE_URL"]


def validate_env() -> None:
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


def install_requirements() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def main() -> None:
    validate_env()
    install_requirements()
    print("Setup complete.")


if __name__ == "__main__":
    main()
