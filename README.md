# Гібридне оцінювання відповідності резюме вакансії (семантичні вектори + правила)

Відтворюваний артефакт практичних робіт з дисципліни «Методологія та організація наукових досліджень»
(ХНУ, КНм-26, Кузьмук В. І.). Розвиток методу CVMaster з бакалаврської роботи автора: відкриті моделі,
публічний бенчмарк, абляція та статистична перевірка.

## Що досліджується
- **RQ1:** чи покращує детермінований шар правил ранжування вакансій порівняно із суто семантичним зіставленням?
- **Бенчмарк:** Vanetik & Kogan (2023), *Information* 14(8):468, doi:10.3390/info14080468 — 5 вакансій, 65 резюме,
  ранжування двох анотаторів для CV 1–30 (ліцензія GPL-3.0, `data/raw/LICENSE_DATA_GPLv3`).
- **Метрики:** Spearman ρ, NDCG@5, Top-1 (на кожне резюме), затримка (мс/пару).

## Результати (30 розмічених CV)
| Конфігурація | ρ | NDCG@5 | Top-1 |
|---|---|---|---|
| M0: TF-IDF (лексичний пошук) | 0.062 | 0.881 | 0.400 |
| A: гібрид з евристичними параметрами (α = 1) | −0.193 | 0.825 | 0.200 |
| B: калібрований (α = 0, аспекти 0.5/0.3/0.2) | **0.202** | **0.889** | 0.233 |

Узгодженість двох анотаторів ρ = 0.171, тобто вона обмежує досяжну якість.

## Відтворення в одну команду
```bash
git clone <URL цього репозиторію> && cd cvmatch
dvc pull || ./scripts/get_data.sh  # резюме під контролем DVC (або завантаження з репозиторію авторів датасету)
docker compose up --build          # MLflow UI: http://localhost:5000
```
Без Docker:
```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.12
pip install -r requirements.txt
python train.py                                    # логи в sqlite:///mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
jupyter nbconvert --execute --to notebook notebooks/PR4_experiment_plan.ipynb notebooks/PR5_statistics.ipynb
```
Дані резюме версіонуються через DVC (`data/raw/CV.dvc`): `dvc pull` після налаштування remote.

## Структура
```
src/cvmatch/   data.py · lexicon.py · method.py · metrics.py · experiment.py
notebooks/     PR4_experiment_plan.ipynb · PR5_statistics.ipynb
data/raw/      бенчмарк Vanetik & Kogan (CV під контролем DVC)
train.py       запуск конфігурацій M0/A/B з трекінгом MLflow
Dockerfile · docker-compose.yml · requirements.txt · zenodo.json · CITATION.cff · LICENSE (MIT)
```
## Середовище
Python 3.12/3.13, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, spaCy 3.8.16 + en_core_web_md 3.8.0, MLflow 3.17.1; seed = 42.
