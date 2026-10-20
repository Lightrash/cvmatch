from build_nb import build
C=[]; md=lambda s:C.append(("md",s)); code=lambda s:C.append(("code",s))
md("""# Практична робота 4. План обчислювального експерименту, бенчмарк та абляційне дослідження
**Тема:** гібридне оцінювання відповідності резюме вакансії на основі семантичних векторів і правил.
**Виконав:** Кузьмук В. І., КНм-26. **Бенчмарк:** Vanetik & Kogan (2023), *Information* 14(8):468 — 5 вакансій, 65 резюме, ранжування двох анотаторів для CV 1–30.""")
code("""import sys, json, time, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../src')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from cvmatch.experiment import *
from cvmatch.method import vacancy_model, cv_years
SEED = 42; np.random.seed(SEED)            # фіксація завадового фактора
sns.set_theme(style='whitegrid')
print(json.dumps(env_info(), indent=1, ensure_ascii=False))""")
md("""## Етап 1. Специфікація експерименту
| Тип змінної | Змінна | Рівні / одиниці |
|---|---|---|
| Незалежна X1 | Векторне подання тексту | TF-IDF (лексичне), LSA-50 (латентно-семантичне), spaCy en_core_web_md (word vectors, 300-d) |
| Незалежна X2 | Сила шару правил α (Модуль 2) | 0 (вимкнено), 0.5, 1.0 |
| Залежна Y1 | Якість ранжування вакансій для кожного CV | Spearman ρ, NDCG@5, Top-1 |
| Залежна Y2 | Часові витрати | затримка, мс на пару; час підготовки (векторизації), с |
| Завадові | seed (LSA), розбіжність анотаторів, якість парсингу DOCX, фонові процеси | seed=42; консенсус двох анотаторів; однакове середовище |""")
md("## Етап 2. Підготовка та валідація датасету")
code("""v, cvs, ann, cons, corpus = setup('../data/raw')
print('Вакансій:', len(v), ' резюме:', len(cvs), ' розмічених CV:', len(LABELED))
print(v[['vac_idx','job_title']].to_string(index=False))
bad = ann[~ann.valid].groupby(['annotator','cv_id']).size().index.tolist()
print('Невалідні масиви ранжувань (дублікати позицій):', bad, '-> для цих CV використовується лише другий анотатор')
vm = pd.DataFrame([dict(vac=j, n_critical=len(m['critical']), n_core=len(m['core']), n_optional=len(m['optional']),
       min_years=m['min_years']) for j, m in ((j, vacancy_model(t)) for j, t in zip(v.vac_idx, v.job_description))])
vm""")
code("""# Узгодженість анотаторів (межа досяжної якості): Spearman між ранжуваннями для кожного CV
a = ann[ann.valid].pivot_table(index=['cv_id','vac_idx'], columns='annotator', values='position').dropna()
iaa = pd.Series({i: spearmanr(g[1], g[2]).correlation for i, g in a.groupby(level=0)})
print(f'Inter-annotator agreement: ρ = {iaa.mean():.3f} ± {iaa.std():.3f} (n = {len(iaa)} CV)')
print('Середня консенсусна позиція вакансії (1 = найкраща):', cons.groupby('vac_idx').pos.mean().round(2).to_dict())""")
code("""# Стратифіковане розбиття 30 розмічених CV на калібрувальну (80%) та валідаційну (20%) підвибірки
best = cons.loc[cons.groupby('cv_id').pos.idxmin()].set_index('cv_id').vac_idx
strata = best.reindex(LABELED).map(lambda j: j if j in (1, 2) else 0)   # 3 страти: вак.1, вак.2, інші
cal_ids, val_ids = train_test_split(LABELED, test_size=0.2, stratify=strata, random_state=SEED)
cal_ids, val_ids = sorted(cal_ids), sorted(val_ids)
print('Калібрування:', len(cal_ids), cal_ids); print('Валідація:', len(val_ids), val_ids)
# Контроль витоку даних: словник TF-IDF/LSA навчається лише на НЕРОЗМІЧЕНИХ CV (31–65) і вакансіях
labeled_texts = set(cvs[cvs.cv_id.isin(LABELED)].text)
assert not labeled_texts & set(corpus), 'витік: розмічені CV у корпусі векторизатора'
print('Корпус векторизатора:', len(corpus), 'документів; перетин із розміченими CV: 0 -> витоку немає')""")
md("## Етап 3. Базова модель (Baseline M0) — лексичний пошук TF-IDF")
code("""base_cfg = dict(kind='tfidf', use_aspects=False, use_rules=False)
d0, m0, _ = evaluate(base_cfg, v, cvs, cons, corpus, val_ids)
d0_all, _, _ = evaluate(base_cfg, v, cvs, cons, corpus)
print('Validation:', d0[['spearman','ndcg5','top1']].mean().round(4).to_dict())
print('Усі 30 CV:  ', d0_all[['spearman','ndcg5','top1','latency_ms','t_prep_s']].mean().round(4).to_dict())""")
md("""## Етап 4. Абляційне дослідження
Модуль 1 — аспектна декомпозиція (загальний текст / навички / досвід, ваги 0.5/0.3/0.2, як у CVMaster).
Модуль 2 — детермінований шар правил (покриття навичок, досвід, штраф за критичні прогалини та розмиті вимоги).""")
code("""configs = {
  'Baseline (M0)': (base_cfg, 'TF-IDF, загальний текст'),
  'Full (M1)':     (dict(kind='spacy', use_aspects=True,  use_rules=True),  'spaCy + Модуль 1 + Модуль 2'),
  'No-Mod1 (M2)':  (dict(kind='spacy', use_aspects=False, use_rules=True),  'Full без аспектної декомпозиції'),
  'No-Mod2 (M3)':  (dict(kind='spacy', use_aspects=True,  use_rules=False), 'Full без шару правил'),
}
rows = []
for name, (cfg, desc) in configs.items():
    dv, _, _ = evaluate(cfg, v, cvs, cons, corpus, val_ids)
    da, m, _ = evaluate(cfg, v, cvs, cons, corpus)
    rows.append({'Конфігурація': name, 'Фактори': desc, 'T_prep, с': da.t_prep_s.iloc[0], 'Затримка, мс/пару': m.latency_ms,
                 'ρ (Val)': dv.spearman.mean(), 'ρ (30 CV)': da.spearman.mean(), 'NDCG@5 (30 CV)': da.ndcg5.mean(), 'Top-1 (30 CV)': da.top1.mean()})
abl = pd.DataFrame(rows)
ref = abl.loc[abl['Конфігурація']=='Full (M1)', 'ρ (30 CV)'].item()
abl['Δρ відносно Full'] = abl['ρ (30 CV)'] - ref
abl.to_csv('../results/pr4_ablation.csv', index=False); abl.round(4)""")
md("## Етап 4б. Калібрування α та ваг аспектів на калібрувальній підвибірці (24 CV) → перевірка на валідаційній (6 CV)")
code("""grid = []
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    for w in [(1,0,0), (0.5,0.3,0.2), (0.34,0.33,0.33), (0.3,0.5,0.2), (0.3,0.2,0.5)]:
        dc, _, _ = evaluate(dict(kind='spacy', use_aspects=True, use_rules=alpha>0, weights=w, alpha=alpha), v, cvs, cons, corpus, cal_ids)
        grid.append(dict(alpha=alpha, weights=str(w), rho_cal=dc.spearman.mean()))
grid = pd.DataFrame(grid).sort_values('rho_cal', ascending=False); grid.to_csv('../results/pr4_calibration_grid.csv', index=False)
best_cfg = grid.iloc[0]; print('Найкраща конфігурація на калібруванні:', best_cfg.to_dict())
w_best = tuple(float(x) for x in best_cfg.weights.strip('()').split(','))
dv_best, _, _ = evaluate(dict(kind='spacy', use_aspects=True, use_rules=best_cfg.alpha>0, weights=w_best, alpha=best_cfg.alpha), v, cvs, cons, corpus, val_ids)
dv_def, _, _ = evaluate(dict(kind='spacy'), v, cvs, cons, corpus, val_ids)
print(f'Validation ρ: калібрований = {dv_best.spearman.mean():.3f}; евристичні параметри CVMaster (α=1, 0.5/0.3/0.2) = {dv_def.spearman.mean():.3f}')
grid.head(8).round(4)""")
md("## Етап 5 / Самостійна робота. Повний факторний план 3 × 3 (подання × сила правил α)")
code("""fact = []
for kind in ['tfidf', 'lsa', 'spacy']:
    for alpha in [0, 0.5, 1.0]:
        d, m, _ = evaluate(dict(kind=kind, use_aspects=True, use_rules=alpha>0, alpha=alpha), v, cvs, cons, corpus)
        fact.append(dict(representation=kind, alpha=alpha, rho=d.spearman.mean(), rho_sd=d.spearman.std(),
                         ndcg5=d.ndcg5.mean(), top1=d.top1.mean(), t_prep_s=d.t_prep_s.iloc[0], latency_ms=m.latency_ms))
fact = pd.DataFrame(fact); fact.to_csv('../results/pr4_factorial.csv', index=False); fact.round(4)""")
code("""fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for kind, mk in zip(['tfidf','lsa','spacy'], ['o','s','^']):
    d = fact[fact.representation==kind]
    ax[0].plot(d.alpha, d.rho, marker=mk, label=kind); ax[1].plot(d.alpha, d.ndcg5, marker=mk, label=kind)
ax[0].axhline(iaa.mean(), ls='--', c='grey', lw=1, label='узгодженість анотаторів')
ax[0].set(xlabel='Сила шару правил α', ylabel='Spearman ρ (30 CV)', title='Якість ранжування')
ax[1].set(xlabel='Сила шару правил α', ylabel='NDCG@5', title='NDCG@5')
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig('../figures/pr4_factorial.png', dpi=200); plt.show()""")
code("""# Шаблонний модуль автоматичної фіксації часових та обчислювальних витрат
import contextlib, resource
@contextlib.contextmanager
def track(name, log):
    t0, c0 = time.perf_counter(), time.process_time()
    yield
    log.append(dict(stage=name, wall_s=round(time.perf_counter()-t0, 4), cpu_s=round(time.process_time()-c0, 4),
                    max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1)))
log = []
with track('load data', log): _ = setup('../data/raw')
with track('evaluate Full (30 CV x 5 vacancies)', log): evaluate(dict(kind='spacy'), v, cvs, cons, corpus)
pd.DataFrame(log)""")
build(C, "PR4_experiment_plan.ipynb")
