#!/home/hswe_junas/projects/LLaMA-Factory/.venv/bin/python3
import sys

from llamafactory.cli_custom import main


if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
