"""Compatibility wrapper for the specification's model export stage."""
from pathlib import Path
import runpy
if __name__=='__main__': runpy.run_path(str(Path(__file__).with_name('15_export_model.py')),run_name='__main__')
