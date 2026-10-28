"""Compatibility wrapper for the primary MACD evaluation."""
from pathlib import Path
import runpy
if __name__=='__main__': runpy.run_path(str(Path(__file__).with_name('04_evaluate_macd.py')),run_name='__main__')
