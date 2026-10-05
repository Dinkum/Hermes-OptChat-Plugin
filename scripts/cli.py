"""Invoke the actual Hermes CLI in the isolated development environment."""
import sys
from hermes_cli.main import main

sys.argv = ["hermes",*sys.argv[1:]]
raise SystemExit(main())
