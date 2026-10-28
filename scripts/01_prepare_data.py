"""WP1: Prepare cleaned, labelled session records from the PhysioNet archive."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loading import load_sessions,save_sessions
def main():
    sessions=load_sessions(); save_sessions(sessions); print(f'Saved {len(sessions)} sessions from {len({s.subject_id for s in sessions})} subjects.')
if __name__=='__main__': main()
