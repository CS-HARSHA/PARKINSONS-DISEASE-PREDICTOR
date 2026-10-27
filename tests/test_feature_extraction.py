from src.data_loading import Session
from src.feature_extraction import session_feature_rows, subject_feature_rows

def test_session_and_subject_rows_have_expected_schema():
    sessions = [Session('1', 'MIT-CS1PD', '1_1', 1, [1, 2, 3]), Session('1', 'MIT-CS1PD', '1_2', 1, [2, 3, 4])]
    session_rows = session_feature_rows(sessions, feature_fn=lambda x: float(x.mean()))
    subject_rows = subject_feature_rows(sessions, feature_fn=lambda x: float(x.mean()))
    assert session_rows[0]['n_keystrokes'] == 3
    assert subject_rows[0]['n_sessions'] == 2
    assert subject_rows[0]['macd'] == 2.5
