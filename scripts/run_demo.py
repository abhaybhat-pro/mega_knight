"""Pure-Python demo entry. scripts/run_demo.sh only exports a throwaway HMAC key."""
import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).with_name("demo_pipeline.py")
sys.argv[0] = str(TARGET)
raise SystemExit(runpy.run_path(str(TARGET), run_name="__main__"))
