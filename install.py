"""Install OptChat through the existing Hermes runtime."""
import sys
from pathlib import Path

from scripts import hermes_python


def main():
    root = Path(__file__).resolve().parent
    sys.argv = [str(root/"scripts/hermes_python.py"), "--installed-home",
                str(root/"scripts/install.py"), *sys.argv[1:]]
    hermes_python.main()


if __name__ == "__main__":
    main()
