"""WP1: Discover PhysioNet cohorts, join labels, and build shared Session records."""
from __future__ import annotations
import csv, json, zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from . import config
from .cleaning import clean_session_events
@dataclass(init=False)
class Session:
    session_id:str; subject_id:str; cohort:str; label:int; hold_times_s:list[float]; press_times_s:list[float|None]; n_keystrokes:int; source_file:str
    def __init__(self, session_id, subject_id, cohort, label, hold_times_s, press_times_s=None, n_keystrokes=None, source_file=''):
        # Accept the legacy positional order (subject_id, cohort, session_id, ...)
        # used by the original unit tests.
        if isinstance(subject_id, str) and subject_id in config.COHORTS.values():
            session_id, subject_id, cohort = cohort, session_id, subject_id
        self.session_id=str(session_id); self.subject_id=str(subject_id); self.cohort=str(cohort); self.label=int(label)
        self.hold_times_s=list(hold_times_s); self.press_times_s=list(press_times_s) if press_times_s is not None else [None]*len(self.hold_times_s)
        self.n_keystrokes=int(n_keystrokes if n_keystrokes is not None else len(self.hold_times_s)); self.source_file=str(source_file)
    @property
    def hold_times(self): return self.hold_times_s
    @property
    def press_times(self): return self.press_times_s

def extract_archive(raw_dir:Path=config.RAW_DIR)->Path:
    candidates=sorted(raw_dir.glob('*.zip'))
    if not candidates: raise FileNotFoundError(f'No PhysioNet zip found in {raw_dir}')
    marker=raw_dir/'.extracted'
    if not marker.exists():
        with zipfile.ZipFile(candidates[0]) as zf: zf.extractall(raw_dir)
        marker.touch()
    return raw_dir

def _find_cohort_dir(raw_dir, cohort):
    matches=[p for p in raw_dir.rglob(cohort) if p.is_dir()]
    if not matches: raise FileNotFoundError(f'Could not find cohort directory {cohort}')
    return matches[0]
def _read_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def _find_ground_truth(cohort_dir, cohort):
    matches=list(cohort_dir.rglob(f'GT_DataPD_{cohort}.csv'))
    if not matches: raise FileNotFoundError(f'Ground-truth CSV not found below {cohort_dir}')
    return matches[0]
def _parse_file_names(row):
    return [v.strip() for k,v in row.items() if k.lower().startswith('file') and v and v.strip()]
def load_sessions(raw_dir:Path=config.RAW_DIR):
    extract_archive(raw_dir); sessions=[]; audit=[]
    for cohort_code in config.COHORTS.values():
        cohort_dir=_find_cohort_dir(raw_dir, cohort_code); gt=_find_ground_truth(cohort_dir, cohort_code)
        data_dir=cohort_dir/f'data_{cohort_code}'
        if not data_dir.exists(): data_dir=cohort_dir
        for row in _read_rows(gt):
            subject_id=str(row['pID']).strip(); label=int(str(row['gt']).strip().lower()=='true')
            for idx, name in enumerate(_parse_file_names(row),1):
                path=data_dir/Path(name).name
                hold, press, summary=clean_session_events(path)
                sid=f'{subject_id}_{idx}'
                audit.append({'subject_id':subject_id,'cohort':cohort_code,'session_id':sid,'source_file':str(path.relative_to(raw_dir)),**asdict(summary)})
                if len(hold)<config.MIN_SESSION_KEYSTROKES: continue
                sessions.append(Session(sid,subject_id,cohort_code,label,hold.tolist(),press.tolist(),int(len(hold)),str(path.relative_to(raw_dir))))
    config.PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    (config.PROCESSED_DIR/'cleaning_summary.json').write_text(json.dumps(audit,indent=2))
    return sessions
def save_sessions(sessions, output_dir=config.PROCESSED_DIR):
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/'sessions.json').write_text(json.dumps([asdict(s) for s in sessions],indent=2))
def load_processed(path=config.PROCESSED_DIR/'sessions.json'):
    rows=json.loads(Path(path).read_text()); out=[]
    for r in rows:
        if 'hold_times_s' not in r:
            r={'session_id':r['session_id'],'subject_id':r['subject_id'],'cohort':r['cohort'],'label':r['label'],'hold_times_s':r.get('hold_times',[]),'press_times_s':r.get('press_times', [None]*len(r.get('hold_times',[]))),'n_keystrokes':len(r.get('hold_times',[])),'source_file':r.get('source_file','')}
        out.append(Session(**r))
    return out
