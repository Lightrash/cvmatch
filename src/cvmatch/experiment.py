"""Допоміжні функції обчислювального експерименту."""
import platform, os, time
import numpy as np, pandas as pd
from .data import load_vacancies, load_cvs, load_annotations, consensus
from .method import Representation, HybridMatcher
from .metrics import per_cv_metrics

LABELED = list(range(1, 31))          # CV з еталонними ранжуваннями
SEED = 42

def setup(raw="data/raw"):
    v, cvs, ann = load_vacancies(raw), load_cvs(raw), load_annotations(raw)
    cons = consensus(ann)
    unl = cvs[~cvs.cv_id.isin(LABELED)]
    corpus = list(unl.text) + list(v.job_description)   # корпус для TF-IDF/LSA: лише нерозмічені CV + вакансії
    return v, cvs, ann, cons, corpus

_REP = {}
def rep(kind, corpus, seed=SEED):
    key = (kind, seed)
    if key not in _REP: _REP[key] = Representation(kind, seed=seed).fit(corpus)
    return _REP[key]

def evaluate(cfg, v, cvs, cons, corpus, ids=LABELED):
    """cfg: kind, use_aspects, use_rules, weights, alpha. Повертає (per-CV DataFrame, matcher, scores)."""
    cfg = dict(dict(kind="spacy", use_aspects=True, use_rules=True, weights=(0.5, 0.3, 0.2), alpha=1.0), **cfg)
    t0 = time.perf_counter()
    m = HybridMatcher(rep(cfg["kind"], corpus), cfg["use_aspects"], cfg["use_rules"], cfg["weights"], cfg["alpha"]).prepare(cvs, v)
    t_prep = time.perf_counter() - t0
    S = m.score_matrix(ids)
    df = pd.DataFrame(per_cv_metrics(S, cons, ids)); df["latency_ms"] = m.latency_ms; df["t_prep_s"] = t_prep
    return df, m, S

def env_info():
    import sklearn, scipy, spacy
    return {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "n_cpu": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__, "spacy": spacy.__version__, "spacy_model": "en_core_web_md-3.8.0"}
