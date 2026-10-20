"""Гібридний метод: семантичні вектори (3 аспекти) + детерміновані правила калібрування.
Модуль 1 — аспектна декомпозиція (загальний текст / навички / досвід, ваги 0.5/0.3/0.2).
Модуль 2 — шар правил: покриття критичних/основних/додаткових навичок, відповідність досвіду,
штраф за відсутні критичні навички (параметри з method-config.json CVMaster)."""
import re, time, datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .lexicon import CANON, TERMS, REL_OF, CRITICAL_MARKERS, OPTIONAL_MARKERS

TIERS = dict(exact=1.0, related=0.5)               # matchSkillTiers (синоніми зводяться до канону)
PENALTY_PER_CRIT, MAX_CRIT_PENALTY = 0.12, 0.36    # matchPenalties
MATCH_W = dict(critical=0.42, core=0.24, optional=0.06, experience=0.18)  # matchWeights (без levelFit)
CUR_YEAR = 2023
VAGUE_MAX, VAGUE_FULL = 0.30, 12                # макс. штраф та кількість вимог для «повної» вакансії                                    # рік створення датасету

_term_re = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in TERMS) + r")(?![a-z0-9])")

def extract_skills(text):
    return {CANON[m] for m in _term_re.findall(text.lower())}

def split_sections(cv):
    """Грубе виділення розділів резюме: навички та досвід (за заголовками)."""
    low = cv.lower(); lines = cv.split("\n"); skills, exp, cur = [], [], None
    for ln in lines:
        l = ln.lower().strip()
        if re.match(r"^(professional |technical |key )?skills|^technologies|^technical", l): cur = "s"
        elif re.match(r"^(professional |work )?experience|^employment|^work history|^career", l): cur = "e"
        elif re.match(r"^education|^languages|^courses|^certific|^military|^personal", l): cur = None
        if cur == "s": skills.append(ln)
        elif cur == "e": exp.append(ln)
    s = "\n".join(skills) or " ".join(sorted(extract_skills(cv)))
    e = "\n".join(exp) or cv
    return s, e

def cv_years(cv):
    """Досвід, років: явна згадка 'N+ years' або сума інтервалів 'YYYY - YYYY/now'."""
    low = cv.lower()
    explicit = [int(x) for x in re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)", low) if int(x) < 40]
    spans = re.findall(r"((?:19|20)\d{2})\s*[-–—]+\s*((?:19|20)\d{2}|now|present|current|today)", low)
    tot = 0
    for a, b in spans:
        b = CUR_YEAR if not b[0].isdigit() else int(b)
        if 0 <= b - int(a) < 40: tot += b - int(a)
    return max(max(explicit, default=0), min(tot, 40))

def vacancy_model(text):
    """Структурована модель вакансії: критичні / основні / додаткові навички, мін. досвід."""
    sents = re.split(r"(?<=[.;!?])\s+|\n", text)   # двокрапка не розриває речення
    crit, opt, core = set(), set(), set()
    for s in sents:
        sk = extract_skills(s); l = s.lower()
        if any(m in l for m in OPTIONAL_MARKERS): opt |= sk
        elif any(m in l for m in CRITICAL_MARKERS): crit |= sk
        else: core |= sk
    core -= crit; opt -= crit | core
    yrs = [int(x) for x in re.findall(r"(\d{1,2})\+?\s*(?:-\s*\d+\s*)?(?:years|yrs)", text.lower()) if int(x) < 20]
    return dict(critical=crit, core=core, optional=opt, min_years=min(yrs) if yrs else 0,
                n_req=len(crit | core | opt))

def coverage(req, have):
    if not req: return None
    sc = []
    for r in req:
        if r in have: sc.append(TIERS["exact"])
        elif REL_OF.get(r, set()) & have: sc.append(TIERS["related"])
        else: sc.append(0.0)
    return float(np.mean(sc))

def rule_layer(vm, cv_sk, years):
    """Повертає детермінований бал [0,1], поправку Δ (в балах 0..100) та пояснення."""
    cov = {b: coverage(vm[b], cv_sk) for b in ("critical", "core", "optional")}
    exp_fit = 1.0 if vm["min_years"] == 0 else min(1.0, years / vm["min_years"])
    parts = {k: v for k, v in dict(critical=cov["critical"], core=cov["core"], optional=cov["optional"],
                                   experience=exp_fit).items() if v is not None}
    wsum = sum(MATCH_W[k] for k in parts)
    s_rule = sum(MATCH_W[k] * v for k, v in parts.items()) / wsum
    missing = sorted(r for r in vm["critical"] if r not in cv_sk and not (REL_OF.get(r, set()) & cv_sk))
    pen = min(MAX_CRIT_PENALTY, PENALTY_PER_CRIT * len(missing))
    # Поправка «неповнота опису вакансії» (CVMaster: flagThresholds / weak requirements):
    # вакансія з малою кількістю явних вимог отримує штраф (інструкція HR: уникати розмитих вимог)
    vague = VAGUE_MAX * (1 - min(1.0, vm["n_req"] / VAGUE_FULL))
    return s_rule, pen + vague, dict(coverage=cov, exp_fit=exp_fit, missing_critical=missing, vague=vague)

class Representation:
    """Векторні подання: 'tfidf' (лексичне), 'lsa' (латентно-семантичне), 'spacy' (word vectors GloVe-подібні)."""
    def __init__(self, kind="spacy", lsa_dim=50, seed=42):
        self.kind, self.lsa_dim, self.seed = kind, lsa_dim, seed
    def fit(self, corpus):
        if self.kind in ("tfidf", "lsa"):
            self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1, ngram_range=(1, 2)).fit(corpus)
            if self.kind == "lsa":
                X = self.vec.transform(corpus)
                self.svd = TruncatedSVD(min(self.lsa_dim, X.shape[1] - 1, X.shape[0] - 1), random_state=self.seed).fit(X)
        else:
            import spacy
            self.nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
        return self
    def embed(self, texts):
        if self.kind == "tfidf": X = self.vec.transform(texts).toarray()
        elif self.kind == "lsa": X = self.svd.transform(self.vec.transform(texts))
        else:
            X = []
            for d in self.nlp.pipe([t[:20000] for t in texts]):
                toks = [t.vector for t in d if t.has_vector and not t.is_stop and t.is_alpha]
                X.append(np.mean(toks, 0) if toks else np.zeros(300))
            X = np.array(X)
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
        return X / n                                   # L2-нормалізація (як у CVMaster)

def sim01(a, b):
    return float(np.clip((a @ b + 1) / 2 if a.min() < 0 or b.min() < 0 else a @ b, 0, 1))

class HybridMatcher:
    def __init__(self, rep, use_aspects=True, use_rules=True, weights=(0.5, 0.3, 0.2), alpha=1.0):
        self.rep, self.use_aspects, self.use_rules, self.w, self.alpha = rep, use_aspects, use_rules, weights, alpha
    def prepare(self, cvs, vacs):
        self.cv_txt = dict(zip(cvs.cv_id, cvs.text)); self.vac_txt = dict(zip(vacs.vac_idx, vacs.job_description))
        cv_ids, vac_ids = list(self.cv_txt), list(self.vac_txt)
        sec = {i: split_sections(t) for i, t in self.cv_txt.items()}
        vm = {j: vacancy_model(t) for j, t in self.vac_txt.items()}
        E = lambda L: dict(zip(L[0], self.rep.embed(L[1])))
        self.E_cv = E((cv_ids, [self.cv_txt[i] for i in cv_ids]))
        self.E_cs = E((cv_ids, [sec[i][0] for i in cv_ids])); self.E_ce = E((cv_ids, [sec[i][1] for i in cv_ids]))
        self.E_v = E((vac_ids, [self.vac_txt[j] for j in vac_ids]))
        self.E_vs = E((vac_ids, [" ".join(sorted(vm[j]["critical"] | vm[j]["core"] | vm[j]["optional"])) or self.vac_txt[j] for j in vac_ids]))
        self.E_ve = self.E_v
        self.vm, self.cv_sk = vm, {i: extract_skills(t) for i, t in self.cv_txt.items()}
        self.yrs = {i: cv_years(t) for i, t in self.cv_txt.items()}
        return self
    def score(self, i, j):
        so = sim01(self.E_cv[i], self.E_v[j])
        if self.use_aspects:
            ss, se = sim01(self.E_cs[i], self.E_vs[j]), sim01(self.E_ce[i], self.E_ve[j])
            s_neural = 100 * (self.w[0] * so + self.w[1] * ss + self.w[2] * se) / sum(self.w)
        else:
            s_neural = 100 * so
        expl = {}
        if self.use_rules:
            s_rule, pen, expl = rule_layer(self.vm[j], self.cv_sk[i], self.yrs[i])
            # Δ: зсув до детермінованого бала + штраф за критичні прогалини (масштаб alpha)
            delta = self.alpha * (100 * (s_rule - 0.5) * 0.5 - 100 * pen)
            s = float(np.clip(s_neural + delta, 0, 100))
        else:
            s = s_neural
        return s, s_neural, expl
    def score_matrix(self, cv_ids):
        t0 = time.perf_counter()
        S = {(i, j): self.score(i, j)[0] for i in cv_ids for j in self.vac_txt}
        self.latency_ms = 1000 * (time.perf_counter() - t0) / len(S)
        return S
