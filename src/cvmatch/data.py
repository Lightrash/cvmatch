"""Завантаження бенчмарку Vanetik & Kogan (2023): 5 вакансій, 65 резюме, ранжування 2 анотаторів для CV 1-30."""
import os, re, ast
import numpy as np, pandas as pd, docx

def _read_docx(path):
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for r in t.rows:
            parts.append(" ".join(c.text for c in r.cells if c.text.strip()))
    return "\n".join(parts)

def load_vacancies(raw="data/raw"):
    v = pd.read_csv(os.path.join(raw, "5_vacancies.csv"))
    v["vac_idx"] = np.arange(1, len(v) + 1)          # номер вакансії 1..5 (порядок у файлі)
    return v

def load_cvs(raw="data/raw"):
    rows = []
    for f in os.listdir(os.path.join(raw, "CV")):
        if f.endswith(".docx"):
            rows.append(dict(cv_id=int(f.split(".")[0]), text=_read_docx(os.path.join(raw, "CV", f))))
    return pd.DataFrame(rows).sort_values("cv_id").reset_index(drop=True)

def load_annotations(raw="data/raw"):
    """Повертає DataFrame: cv_id, annotator, vac_idx, position (1 = найкраща), valid.
    Масив анотатора – впорядкований список номерів вакансій від найкращої до найгіршої.
    Масиви з дублікатами (невалідні перестановки) позначаються valid=False."""
    txt = open(os.path.join(raw, "annotations-for-the-first-30-vacancies.txt"), encoding="utf-8").read()
    out = []
    for a in (1, 2):
        m = re.search(rf"ANNOTATOR_{a}_RANKINGS\s*=\s*(\[.*?\]\])", txt, re.S)
        arr = ast.literal_eval(re.sub(r"#[^\n]*", "", m.group(1)))
        for i, order in enumerate(arr, start=1):
            valid = sorted(order) == [1, 2, 3, 4, 5]
            for pos, vac in enumerate(order, start=1):
                if valid:
                    out.append(dict(cv_id=i, annotator=a, vac_idx=vac, position=pos, valid=True))
            if not valid:
                for vac in range(1, 6):
                    out.append(dict(cv_id=i, annotator=a, vac_idx=vac, position=np.nan, valid=False))
    return pd.DataFrame(out)

def consensus(ann):
    """Консенсусна позиція = середнє валідних позицій анотаторів; релевантність rel = 5 - позиція (0..4)."""
    g = ann[ann.valid].groupby(["cv_id", "vac_idx"]).position.mean().rename("pos").reset_index()
    g["rel"] = 5 - g["pos"]
    return g
