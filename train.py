"""Відтворюваний запуск експерименту з трекінгом MLflow.
Запуск: python train.py  (або docker compose up)"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import mlflow
from cvmatch.experiment import setup, evaluate, env_info, LABELED, SEED

CONFIGS = {
    "M0_tfidf_baseline":  dict(kind="tfidf", use_aspects=False, use_rules=False),
    "A_heuristic_hybrid": dict(kind="spacy", use_aspects=True, use_rules=True, weights=(0.5, 0.3, 0.2), alpha=1.0),
    "B_calibrated":       dict(kind="spacy", use_aspects=True, use_rules=False, weights=(0.5, 0.3, 0.2), alpha=0.0),
}

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("cv-vacancy-matching")
    v, cvs, ann, cons, corpus = setup(os.environ.get("DATA_DIR", "data/raw"))
    os.makedirs("results", exist_ok=True)
    for name, cfg in CONFIGS.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v_) for k, v_ in cfg.items()} | {"seed": SEED, "n_eval_cv": len(LABELED), "dataset": "Vanetik&Kogan2023"})
            mlflow.set_tags(env_info())
            d, m, S = evaluate(cfg, v, cvs, cons, corpus)
            mlflow.log_metrics({"spearman_mean": d.spearman.mean(), "spearman_std": d.spearman.std(),
                                "ndcg5_mean": d.ndcg5.mean(), "top1_rate": d.top1.mean(),
                                "latency_ms_per_pair": m.latency_ms, "t_prep_s": float(d.t_prep_s.iloc[0])})
            path = f"results/{name}_per_cv.csv"; d.to_csv(path, index=False); mlflow.log_artifact(path)
            with open(f"results/{name}_config.json", "w") as f: json.dump(cfg, f, indent=1, default=str)
            mlflow.log_artifact(f"results/{name}_config.json")
            print(f"{name:20s} rho={d.spearman.mean():+.4f} ndcg5={d.ndcg5.mean():.4f} top1={d.top1.mean():.3f}")
