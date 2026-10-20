"""Evaluate saved models on seen-test, unseen generators, degraded clips; LOGO; metadata baseline."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS, RESULTS, SEED, SEEN, UNSEEN
from src.models import data as D
from src.models.cnn import SpecTempCNN, cnn_predict


def metrics(y, p, thresh: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thresh
    out = {"n": int(len(y)), "acc": float(accuracy_score(y, pred)) if len(y) else float("nan")}
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    out["f1"] = float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan")
    out["tpr_fake"] = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    out["tnr_real"] = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return out


def load_models(tag: str = "") -> dict:
    ms = {}
    for p in sorted(MODELS.glob(f"*{tag}.joblib")):
        name = p.stem
        if tag == "" and "_logo_" in name:
            continue
        ms[name] = joblib.load(p)
    cp = MODELS / f"cnn{tag}.pt"
    if cp.exists():
        ck = torch.load(cp, weights_only=False)
        m = SpecTempCNN(ck["in_ch"], ck["n_bins"])
        m.load_state_dict(ck["state"])
        ms[f"cnn{tag}"] = {"cnn": m, "stats": (ck["mu"], ck["sd"])}
    return ms


def predict(entry: dict, df: pd.DataFrame, seq_name: str = "clean", keys=None) -> np.ndarray:
    if "cnn" in entry:
        keys = df.clip_id if keys is None else keys
        return cnn_predict(entry["cnn"], entry["stats"], D.load_seq(seq_name, keys))
    X, _ = D.xy(df, entry["cols"])
    return entry["model"].predict_proba(X)[:, 1]


def _balanced_unseen(test: pd.DataFrame, gen: str) -> pd.DataFrame:
    fake = test[test.generator == gen]
    real = test[(test.generator == "real")]
    real = real.sample(n=min(len(real), len(fake)), random_state=SEED)
    return pd.concat([fake, real])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = D.clean_table()
    test_all = df[(df.split == "test") & ~df.aug.astype(bool)]
    seen_test = test_all[~test_all.generator.isin(UNSEEN)]
    models = load_models()
    rows, gen_rows, preds = [], [], []
    for name, entry in models.items():
        p = predict(entry, seen_test)
        rows.append({"model": name, "set": "seen_test", **metrics(seen_test.label, p)})
        for gen in SEEN + UNSEEN:
            sub = seen_test if gen in SEEN else _balanced_unseen(test_all, gen)
            sub = sub[sub.generator.isin([gen, "real"])]
            if (sub.generator == gen).sum() == 0:
                continue
            pp = predict(entry, sub)
            m = metrics(sub.label, pp)
            gen_rows.append({"model": name, "generator": gen, "seen": gen in SEEN, **m})
        pa = predict(entry, test_all)
        preds.append(pd.DataFrame({"clip_id": test_all.clip_id.values, "model": name, "p_fake": pa,
                                   "label": test_all.label.values, "generator": test_all.generator.values}))
        # all-unseen pooled
        un = pd.concat([_balanced_unseen(test_all, g) for g in UNSEEN if (test_all.generator == g).any()])
        if len(un):
            un = un.drop_duplicates("clip_id")
            rows.append({"model": name, "set": "unseen_test", **metrics(un.label, predict(entry, un))})
    pd.DataFrame(rows).to_csv(RESULTS / "test_metrics.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(RESULTS / "per_generator.csv", index=False)
    pd.concat(preds).to_csv(RESULTS / "predictions.csv", index=False)

    # degraded
    deg_rows = []
    try:
        deg = D.degraded_table()
        deg = deg[(deg.split == "test")]
        for name, entry in models.items():
            for dname, sub in deg.groupby("degradation"):
                for subset, s in (("seen", sub[~sub.generator.isin(UNSEEN)]), ("all", sub)):
                    keys = s.clip_id + "|" + dname
                    p = predict(entry, s, "degraded", keys=keys)
                    deg_rows.append({"model": name, "degradation": dname, "subset": subset, **metrics(s.label, p)})
        pd.DataFrame(deg_rows).to_csv(RESULTS / "degraded.csv", index=False)
    except FileNotFoundError:
        print("no degraded features")

    # metadata-only shortcut baseline (original resolution, fps, duration)
    meta_cols = ["orig_w", "orig_h", "orig_fps", "duration"]
    tr = D.select(df, "train")
    mm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    Xtr = tr[meta_cols].fillna(0).to_numpy()
    mm.fit(Xtr, tr.label)
    meta_res = metrics(seen_test.label, mm.predict_proba(seen_test[meta_cols].fillna(0).to_numpy())[:, 1])
    pd.DataFrame([{"model": "metadata_lr", "set": "seen_test", **meta_res}]).to_csv(
        RESULTS / "metadata_baseline.csv", index=False)

    # leave-one-generator-out
    logo_rows = []
    from src.models.train import train_classic
    from src.models.cnn import train_cnn
    for held in SEEN:
        keep = [g for g in SEEN if g != held]
        sub = df[df.generator.isin(keep + ["real"])]
        m, cols, _ = train_classic(sub, "lgbm", "all")
        tst = test_all[test_all.generator.isin([held, "real"])]
        tst = _balanced_unseen(tst, held)
        p = m.predict_proba(D.xy(tst, cols)[0])[:, 1]
        logo_rows.append({"model": "lgbm_all", "held_out": held, **metrics(tst.label, p)})
        tr2, va2 = D.select(sub, "train", include_aug=True), D.select(sub, "val")
        cm, st, _ = train_cnn(D.load_seq("clean", tr2.clip_id), tr2.label.to_numpy(),
                              D.load_seq("clean", va2.clip_id), va2.label.to_numpy())
        logo_rows.append({"model": "cnn", "held_out": held,
                          **metrics(tst.label, cnn_predict(cm, st, D.load_seq("clean", tst.clip_id)))})
        print(logo_rows[-2:], flush=True)
    pd.DataFrame(logo_rows).to_csv(RESULTS / "logo.csv", index=False)

    # headline summary for the app
    t = pd.DataFrame(rows)
    summary = {"seen_test": t[t.set == "seen_test"].set_index("model")[["acc", "auroc", "f1"]].to_dict("index"),
               "unseen_test": t[t.set == "unseen_test"].set_index("model")[["acc", "auroc", "f1"]].to_dict("index"),
               "metadata_baseline": meta_res,
               "counts": df[~df.aug.astype(bool)].groupby(["generator", "split"]).size().unstack(fill_value=0).to_dict("index")}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(t.sort_values(["set", "auroc"], ascending=[True, False]).to_string())


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
