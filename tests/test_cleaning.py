from pathlib import Path
import numpy as np
from src.cleaning import clean_session_csv

def test_cleaning_removes_timestamp_corruption_and_modifiers(tmp_path: Path):
    p = tmp_path / 'session.csv'
    p.write_text('Shift_L,0.9,2.1,1.2\nh,1402930352.8,2.2,-1402930350.6\na,0.1,2.6,2.5\nBackSpace,0.2,2.9,2.7\n')
    values, summary = clean_session_csv(p)
    assert np.allclose(values, [0.1])
    assert summary.removed_invalid == 1
    assert summary.removed_long_modifier == 1
    assert summary.removed_backspace == 1
