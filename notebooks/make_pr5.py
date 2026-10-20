from build_nb import build
C=[]; md=lambda s:C.append(("md",s)); code=lambda s:C.append(("code",s))
md("""# Практична робота 5. Статистична обробка результатів обчислювального експерименту
**Виконав:** Кузьмук В. І., КНм-26.
Порівнюються **Алгоритм A** — гібрид з евристичними параметрами CVMaster (spaCy, аспекти 0.5/0.3/0.2, шар правил α = 1) та
**Алгоритм B** — калібрований варіант: параметри (α, ваги аспектів) обираються на даних за схемою leave-one-out —
для кожного CV конфігурація обирається за іншими 29 CV, тож оцінка B не є оптимістично зміщеною.
Одиниця спостереження — резюме (30 незалежних CV з еталонним ранжуванням 5 вакансій); для часових метрик — 30 повторних прогонів.""")
code("""import sys, time, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../src')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.metrics import (confusion_matrix, classification_report, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from cvmatch.experiment import *
sns.set_theme(style='whitegrid'); ALPHA = 0.05
v, cvs, ann, cons, corpus = setup('../data/raw')
CFG = {'A': dict(kind='spacy', use_aspects=True, use_rules=True, weights=(0.5,0.3,0.2), alpha=1.0),
       'B': dict(kind='spacy', use_aspects=True, use_rules=False, weights=(0.34,0.33,0.33), alpha=0.0),
       'M0': dict(kind='tfidf', use_aspects=False, use_rules=False)}""")
md("## Етап 1. Логи експерименту та описова статистика")
code("""res = {}
for k in ['A', 'M0']:
    d, m, S = evaluate(CFG[k], v, cvs, cons, corpus); d['alg'] = k; res[k] = (d, m, S)
# Leave-one-out калібрування B: сітка α × ваги, вибір за 29 CV, оцінювання на відкладеному CV
GRID = [dict(kind='spacy', use_aspects=True, use_rules=a > 0, alpha=a, weights=w)
        for a in [0, 0.25, 0.5, 0.75, 1.0] for w in [(1,0,0), (0.5,0.3,0.2), (0.34,0.33,0.33), (0.3,0.5,0.2), (0.3,0.2,0.5)]]
G = [evaluate(g, v, cvs, cons, corpus) for g in GRID]
R = np.array([g[0].set_index('cv_id').spearman.reindex(LABELED).values for g in G])      # конфігурації × CV
rowsB, S_B, chosen = [], {}, []
for n, i in enumerate(LABELED):
    others = [x for x in range(len(LABELED)) if x != n]
    c = int(np.argmax(R[:, others].mean(1))); chosen.append(c)
    rowsB.append(G[c][0].set_index('cv_id').loc[i].to_dict() | dict(cv_id=i))
    S_B.update({(i, j): G[c][2][(i, j)] for j in v.vac_idx})
dB = pd.DataFrame(rowsB); dB['alg'] = 'B'
c_mode = max(set(chosen), key=chosen.count); CFG['B'] = GRID[c_mode]
print('Обрані LOO-конфігурації (частота):', pd.Series([str((GRID[c]['alpha'], GRID[c]['weights'])) for c in chosen]).value_counts().to_dict())
res['B'] = (dB, G[c_mode][1], S_B)
q = pd.concat([res['A'][0], res['B'][0]])
# Часова метрика: 30 повторних прогонів скорингу 150 пар для кожного алгоритму
lat = []
for run in range(30):
    for k in 'AB':
        m = res[k][1]; t0 = time.perf_counter(); m.score_matrix(LABELED); lat.append(dict(run=run, alg=k, latency_ms=m.latency_ms))
lat = pd.DataFrame(lat)
q.to_csv('../results/pr5_per_cv.csv', index=False); lat.to_csv('../results/pr5_latency_runs.csv', index=False)
def iqr(x): return x.quantile(.75) - x.quantile(.25)
desc = pd.concat([q.groupby('alg')[['spearman','ndcg5','top1']].agg(['mean','median','std',iqr]),
                  lat.groupby('alg')[['latency_ms']].agg(['mean','median','std',iqr])], axis=1).T.round(4)
desc""")
md("## Етап 2. Перевірка нормальності (Шапіро–Уїлк, Q-Q plot)")
code("""norm = []
for metric, df in [('spearman', q), ('latency_ms', lat)]:
    for alg in 'AB':
        W, p = stats.shapiro(df[df.alg==alg][metric]); norm.append(dict(metric=metric, alg=alg, W=W, p_value=p, normal=p > ALPHA))
    a = df[df.alg=='A'].sort_values(df.columns[0])[metric].values; b = df[df.alg=='B'].sort_values(df.columns[0])[metric].values
    W, p = stats.shapiro(a - b); norm.append(dict(metric=metric, alg='різниця A−B', W=W, p_value=p, normal=p > ALPHA))
norm = pd.DataFrame(norm); norm.round(4)""")
code("""fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
for j, alg in enumerate('AB'):
    x = q[q.alg==alg].spearman
    sns.histplot(x, bins=10, kde=True, ax=ax[0, j], color=['#4C72B0','#DD8452'][j])
    ax[0, j].set(title=f'Алгоритм {alg}: розподіл Spearman ρ по CV', xlabel='ρ')
    stats.probplot(x, dist='norm', plot=ax[1, j]); ax[1, j].set_title(f'Q-Q plot, алгоритм {alg}')
plt.tight_layout(); plt.savefig('../figures/pr5_normality.png', dpi=200); plt.show()""")
md("## Етап 3. Порівняльне статистичне тестування (парні вибірки)")
code("""tests = []
for metric, df, key in [('spearman', q, 'cv_id'), ('ndcg5', q, 'cv_id'), ('latency_ms', lat, 'run')]:
    a = df[df.alg=='A'].sort_values(key)[metric].values; b = df[df.alg=='B'].sort_values(key)[metric].values
    both = norm.query('metric==@metric and alg in [\"A\",\"B\"]').normal.all() if metric in set(norm.metric) else False
    if both: name, (st, p) = 'парний t-критерій Стьюдента', stats.ttest_rel(a, b)
    else:    name, (st, p) = 'критерій Вілкоксона', stats.wilcoxon(a, b)
    dz = (b - a).mean() / (b - a).std(ddof=1)
    boot = [np.mean(np.random.default_rng(i).choice(b - a, len(a))) for i in range(5000)]
    tests.append(dict(metric=metric, test=name, statistic=st, p_value=p, mean_A=a.mean(), mean_B=b.mean(),
                      diff_B_minus_A=(b-a).mean(), ci95_low=np.percentile(boot, 2.5), ci95_high=np.percentile(boot, 97.5),
                      cohen_dz=dz, decision='відхиляємо H0' if p <= ALPHA else 'не відхиляємо H0'))
tests = pd.DataFrame(tests); tests.to_csv('../results/pr5_tests.csv', index=False); tests.round(4)""")
md("""## Етап 4. Матриця помилок і метрики класифікації
Бінаризація: пара «CV–вакансія» **релевантна**, якщо консенсусна позиція ≤ 2 (вакансія у топ-2 для цього CV).
Прогноз: дві вакансії з найвищим балом для кожного CV — «релевантні». Для ROC-AUC бал z-нормалізується всередині CV.""")
code("""def pairs(S):
    P = cons[cons.cv_id.isin(LABELED)].copy(); P['raw'] = [S[(i, j)] for i, j in zip(P.cv_id, P.vac_idx)]
    # бали порівнянні лише в межах одного CV, тому для ROC-AUC використовується z-нормалізація всередині CV
    P['score'] = P.groupby('cv_id').raw.transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    P['y_true'] = (P.pos <= 2).astype(int)
    P['y_pred'] = P.groupby('cv_id').score.rank(ascending=False, method='first').le(2).astype(int)
    return P
PB = pairs(res['B'][2]); y, yp = PB.y_true.values, PB.y_pred.values
cm = confusion_matrix(y, yp); print(pd.DataFrame(cm, index=['нерелев.','релев.'], columns=['прогноз: нерелев.','прогноз: релев.']))
print(classification_report(y, yp, target_names=['нерелевантна','релевантна'], digits=4))
mets = {'Precision (macro)': precision_score(y, yp, average='macro'), 'Recall (macro)': recall_score(y, yp, average='macro'),
        'F1 (macro)': f1_score(y, yp, average='macro'), 'Precision (weighted)': precision_score(y, yp, average='weighted'),
        'Recall (weighted)': recall_score(y, yp, average='weighted'), 'F1 (weighted)': f1_score(y, yp, average='weighted'),
        'ROC-AUC': roc_auc_score(y, PB.score)}
pd.Series(mets).round(4)""")
code("""fig, ax = plt.subplots(figsize=(4.8, 4.2))
ConfusionMatrixDisplay(cm, display_labels=['нерелев.','релев.']).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Матриця помилок, алгоритм B (150 пар)'); plt.tight_layout(); plt.savefig('../figures/pr5_confusion.png', dpi=200); plt.show()""")
md("## Етап 5. Наукова візуалізація")
code("""def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for j, (metric, df, lab) in enumerate([('spearman', q, 'Spearman ρ (на CV)'), ('latency_ms', lat, 'Затримка, мс/пару')]):
    sns.boxplot(data=df, x='alg', y=metric, ax=ax[j], palette=['#4C72B0','#DD8452'], width=.5, order=['A','B'])
    sns.stripplot(data=df, x='alg', y=metric, ax=ax[j], color='k', size=3, alpha=.5, order=['A','B'])
    p = tests.set_index('metric').loc[metric, 'p_value']; y0 = df[metric].max(); h = (df[metric].max() - df[metric].min()) * .08
    ax[j].plot([0,0,1,1], [y0+h*.3, y0+h, y0+h, y0+h*.3], c='k', lw=1)
    ax[j].text(.5, y0+h*1.1, f'{stars(p)}  p = {p:.2e}', ha='center', va='bottom')
    ax[j].set(xticks=[0,1], xticklabels=['A: евристичний гібрид', 'B: калібрований'], xlabel='', ylabel=lab); ax[j].set_ylim(top=y0+h*2.3)
plt.tight_layout(); plt.savefig('../figures/pr5_boxplot.png', dpi=200); plt.show()""")
code("""fig, ax = plt.subplots(figsize=(5, 4.6))
for k, lab, c in [('M0','M0: TF-IDF','#55A868'), ('A','A: евристичний гібрид','#4C72B0'), ('B','B: калібрований','#DD8452')]:
    P = pairs(res[k][2]); fpr, tpr, _ = roc_curve(P.y_true, P.score)
    ax.plot(fpr, tpr, c=c, lw=2, label=f'{lab} (AUC = {roc_auc_score(P.y_true, P.score):.3f})')
ax.plot([0,1], [0,1], 'k--', lw=1, label='Випадкове вгадування (AUC = 0.5)')
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC: релевантність пари CV–вакансія')
ax.legend(loc='lower right', fontsize=8); plt.tight_layout(); plt.savefig('../figures/pr5_roc.png', dpi=200); plt.show()""")
md("""## Самостійна робота. 3 алгоритми × 5 датасетів × 10 прогонів: Фрідман, Немені, Cohen's d
Датасет D_j (j = 1..5) — задача рекрутера «ранжувати 30 кандидатів для вакансії j» (метрика — Spearman ρ між балом і консенсусною релевантністю).
10 прогонів — бутстреп-вибірки 30 CV (seed = 0..9). Разом 50 блоків для тесту Фрідмана.""")
code("""import scikit_posthocs as sp
ALGS = {'M0 (TF-IDF)': 'M0', 'A (гібрид)': 'A', 'B (калібр.)': 'B'}
rel = cons.set_index(['cv_id','vac_idx']).rel
rows = []
for j in v.vac_idx:
    for r in range(10):
        ids = np.random.default_rng(r).choice(LABELED, len(LABELED), replace=True)
        for name, k in ALGS.items():
            S = res[k][2]; s = [S[(i, j)] for i in ids]; t = [rel[(i, j)] for i in ids]
            rows.append(dict(dataset=f'D{j}', run=r, alg=name, rho=stats.spearmanr(s, t).correlation))
cvr = pd.DataFrame(rows); cvr.to_csv('../results/pr5_3algs_5datasets.csv', index=False)
mat = cvr.pivot_table(index='dataset', columns='alg', values='rho')[list(ALGS)]; mat.round(4)""")
code("""blocks = cvr.pivot_table(index=['dataset','run'], columns='alg', values='rho')[list(ALGS)]
chi2, p_fr = stats.friedmanchisquare(*[blocks[a] for a in ALGS])
chi2_5, p_fr5 = stats.friedmanchisquare(*[mat[a] for a in ALGS])
ranks = blocks.rank(axis=1, ascending=False).mean()
print(f'Фрідман (N = 50 блоків): chi2 = {chi2:.3f}, p = {p_fr:.3e}')
print(f'Фрідман (N = 5 датасетів, середні): chi2 = {chi2_5:.3f}, p = {p_fr5:.3e}')
print('Середні ранги (1 = найкращий):'); print(ranks.round(3))
nem = sp.posthoc_nemenyi_friedman(blocks.values); nem.index = nem.columns = list(ALGS)
print('Пост-хок Немені, p-values:'); nem.round(4)""")
code("""from itertools import combinations
eff = []
for a1, a2 in combinations(ALGS, 2):
    diff = blocks[a2] - blocks[a1]; pooled = np.sqrt((blocks[a1].var(ddof=1) + blocks[a2].var(ddof=1)) / 2)
    eff.append(dict(pair=f'{a2} − {a1}', mean_diff=diff.mean(), cohen_d=diff.mean()/pooled, nemenyi_p=nem.loc[a1, a2]))
eff = pd.DataFrame(eff); eff.to_csv('../results/pr5_effects.csv', index=False); eff.round(4)""")
code("""k, N = len(ALGS), len(blocks); CD = 2.343 * np.sqrt(k * (k + 1) / (6 * N))     # q_0.05(k=3) = 2.343 (Демшар, 2006)
fig, ax = plt.subplots(figsize=(7, 1.9)); r = ranks.sort_values()
ax.hlines(0, 1, k, color='k'); ax.set_xlim(.8, k + .2); ax.set_ylim(-1, 1.3); ax.axis('off')
for i, (alg, rk) in enumerate(r.items()):
    ax.plot(rk, 0, 'o', c='k'); ax.annotate(f'{alg}\\n{rk:.2f}', (rk, 0), (rk, .45 + .4*(i % 2)), ha='center', fontsize=9, arrowprops=dict(arrowstyle='-', lw=.6))
ax.plot([1, 1 + CD], [-.5, -.5], c='tab:red', lw=3); ax.text(1 + CD/2, -.9, f'CD = {CD:.3f}', ha='center', color='tab:red')
for x in range(1, k + 1): ax.text(x, -.25, str(x), ha='center', fontsize=8)
plt.tight_layout(); plt.savefig('../figures/pr5_cd.png', dpi=200); plt.show()""")
build(C, "PR5_statistics.ipynb")
