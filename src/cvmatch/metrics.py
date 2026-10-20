import numpy as np
from scipy.stats import spearmanr

def ndcg_at_k(pred_scores, rel, k=5):
    order = np.argsort(-np.asarray(pred_scores))[:k]
    g = np.asarray(rel)[order]; disc = 1 / np.log2(np.arange(2, len(g) + 2))
    ideal = np.sort(rel)[::-1][:k]
    return float((g * disc).sum() / (ideal * disc).sum())

def per_cv_metrics(S, cons, cv_ids):
    """Для кожного CV: Spearman ρ (оцінки vs релевантність), NDCG@5, Top-1 hit."""
    out = []
    for i in cv_ids:
        c = cons[cons.cv_id == i].sort_values("vac_idx")
        s = np.array([S[(i, j)] for j in c.vac_idx]); r = c.rel.values
        rho = spearmanr(s, r).correlation if np.std(s) > 0 else 0.0
        top1 = int(r[np.argmax(s)] == r.max())
        out.append(dict(cv_id=i, spearman=rho, ndcg5=ndcg_at_k(s, r), top1=top1))
    return out
