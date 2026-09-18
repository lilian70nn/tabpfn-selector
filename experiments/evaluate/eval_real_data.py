import torch
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.model.tabpfn import TabularPFNModel
from src.data.collate_real_data_test import collate_openml_task
from experiments.config import CLS_DATASETS, REG_DATASETS


def slice_task_batch(batch, start, end):
    from dataclasses import replace
    B = batch.X_train.shape[0]

    def cut(x):
        if torch.is_tensor(x) and x.ndim > 0 and x.shape[0] == B:
            return x[start:end]
        return x

    return replace(batch, **{name: cut(getattr(batch, name)) for name in batch.__dataclass_fields__})

@torch.no_grad()
def forward_in_chunks(model, batch, chunk_size=5):
    outputs = []
    B = batch.X_train.shape[0]

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        small_batch = slice_task_batch(batch, start, end)
        outputs.append(model(small_batch))

    result = {}
    for key in outputs[0]:
        values = [out[key] for out in outputs]
        result[key] = torch.cat(values, dim=0) if torch.is_tensor(values[0]) else values[0]

    return result


@torch.no_grad()
def evaluate_batch_tasks(model, batch, out, task_kind):
    rows = []
    logits = out["logits"]
    test_mask = out["test_mask"]

    if task_kind == "classification":
        C = logits.shape[-1]
        class_idx = torch.arange(C, device=logits.device)[None, None, :]
        valid_class = class_idx < batch.n_classes[:, None, None]
        logits = logits.masked_fill(~valid_class, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        for b in range(logits.shape[0]):
            mask = test_mask[b]
            yt = batch.y_test[b, mask].long().detach().cpu().numpy()
            yp = preds[b, mask].detach().cpu().numpy()
            prob = probs[b, mask].detach().cpu().numpy()
            c = int(batch.n_classes[b].item())
            labels = list(range(c))

            p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=labels, average="macro", zero_division=0)
            metrics = {
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": float(r),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }

            try:
                if c == 2:
                    metrics["auc"] = float(roc_auc_score(yt, prob[:, 1]))
                else:
                    metrics["auc"] = float(roc_auc_score(yt, prob[:, :c], labels=labels, multi_class="ovr", average="macro"))
            except ValueError:
                metrics["auc"] = np.nan

            rows.append(metrics)

    elif task_kind == "regression":
        borders = model.encoder.regression_borders.to(device=logits.device, dtype=logits.dtype)
        centers = (borders[:-1] + borders[1:]) / 2.0
        centers[0] = borders[1] - (borders[2] - borders[1]) / 2.0
        centers[-1] = borders[-2] + (borders[-2] - borders[-3]) / 2.0

        probs = torch.softmax(logits, dim=-1)
        z_pred = (probs * centers[None, None, :]).sum(dim=-1)
        y_pred = z_pred * batch.y_std[:, None] + batch.y_mean[:, None]

        for b in range(logits.shape[0]):
            mask = test_mask[b]
            yt = batch.y_test[b, mask].float()
            yp = y_pred[b, mask]
            err = yp - yt
            mae = err.abs().mean()
            rmse = torch.sqrt((err ** 2).mean())
            ss_res = ((yt - yp) ** 2).sum()
            ss_tot = ((yt - yt.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else torch.tensor(float("nan"), device=yt.device)
            rows.append({"r2": float(r2), "mae": float(mae), "rmse": float(rmse)})

    else:
        raise ValueError(f"Unknown task_kind: {task_kind}")

    return rows


def get_pfn_importance(batch, out):
    B = out["importance_logits"].shape[0]
    result = []

    for b in range(B):
        d = int(batch.d_emb[b].item())
        imp = torch.softmax(out["importance_logits"][b, :d], dim=-1)
        result.append(imp.detach().cpu().numpy())

    return result


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    rho = spearmanr(a, b).correlation
    return float(rho) if not np.isnan(rho) else float("nan")


def summarize(values):
    values = np.asarray(values, dtype=float)
    return np.nanmean(values), np.nanstd(values, ddof=1)


def topk_indices(imp, k_frac=0.2):
    imp = np.asarray(imp, dtype=float)
    k = max(1, int(np.ceil(k_frac * len(imp))))
    return np.argsort(-imp)[:k]


def evaluate_intervention(model, name, openml_id, selected_features, method, mode, k_frac, task_kind, n_repeats):
    batch = collate_openml_task(
        [(name, openml_id)],
        n_repeats=n_repeats,
        use_selector=True,
        classification=(task_kind == "classification"),
        feature_seed=0,
        split_seed=0,
        shuffle_features=False,
        compute_reference_importance=False,
        reference_seed=0,
        selected_features=selected_features,
    )

    out = forward_in_chunks(model, batch, chunk_size=5)
    metrics_list = evaluate_batch_tasks(model, batch, out, task_kind)
    rows = []

    for b, metrics in enumerate(metrics_list):
        row = {
            "dataset": name,
            "openml_id": openml_id,
            "repeat": b,
            "method": method,
            "mode": mode,
            "k_frac": float(k_frac),
            "n_selected": int(batch.d_emb[b].item()),
            "n_train": int(batch.n_train[b].item()),
            "n_test": int(batch.n_test[b].item()),
            "d_selected": int(batch.d_emb[b].item()),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def print_dataset_keep_remove_diff_tables(all_intervention_rows, name):
    df = pd.DataFrame([r for r in all_intervention_rows if r["dataset"] == name])
    metadata_cols = {"dataset", "openml_id", "repeat", "method", "mode", "k_frac", "n_selected", "n_train", "n_test", "d_selected"}
    metric_cols = [col for col in df.columns if col not in metadata_cols]

    full = df[df["mode"] == "full"][["repeat"] + metric_cols].rename(columns={m: f"full_{m}" for m in metric_cols})
    sub = df[df["mode"] != "full"].merge(full, on="repeat", how="left")
    methods = sub["method"].dropna().unique()

    def make_table(mode):
        rows = []
        for method in methods:
            cur = sub[(sub["method"] == method) & (sub["mode"] == mode)]
            row = {
                "method": method,
                "k_frac": float(cur["k_frac"].iloc[0]) if len(cur) else np.nan,
                "n_selected_mean": float(cur["n_selected"].mean()) if len(cur) else np.nan,
            }
            for metric in metric_cols:
                row[f"{metric}_delta"] = float((cur[metric] - cur[f"full_{metric}"]).mean()) if len(cur) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    keep_table = make_table("keep")
    remove_table = make_table("remove")

    print("\n" + "-" * 120)
    print(f"[{name}] KEEP top-k features only")
    print("Diff is relative to full features.")
    print("delta = intervention_metric - full_metric.")
    print("-" * 120)
    print(keep_table.to_string(index=False))

    print("\n" + "-" * 120)
    print(f"[{name}] REMOVE top-k features")
    print("Diff is relative to full features.")
    print("delta = intervention_metric - full_metric.")
    print("-" * 120)
    print(remove_table.to_string(index=False))
    print("-" * 120)

    return keep_table, remove_table


def main(model, model_path, task_kind="classification", datasets=None, evaluate_importance=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)

    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_repeats = 30
    all_metric_rows = []
    all_imp_rows = []
    all_summary_rows = []
    all_intervention_rows = []

    with torch.no_grad():
        for name, openml_id in datasets.items():
            print("=" * 80)
            print(f"dataset: {name} ({openml_id})")

            batch = collate_openml_task(
                [(name, openml_id)],
                n_repeats=n_repeats,
                use_selector=True,
                classification=(task_kind == "classification"),
                feature_seed=0,
                split_seed=0,
                shuffle_features=True,
                compute_reference_importance=evaluate_importance,
                reference_seed=0,
            )

            out = forward_in_chunks(model, batch, chunk_size=5)
            metrics_list = evaluate_batch_tasks(model, batch, out, task_kind)
            rows = []

            if evaluate_importance:
                pfn_imp_list = get_pfn_importance(batch, out)
                ref_mi_list = []
                ref_rf_list = []
                ref_linear_perm_list = []
                selected_by_method = {"pfn_imp": [], "mi": [], "rf_perm": [], "linear_perm": [], "random": []}
                removed_by_method = {"pfn_imp": [], "mi": [], "rf_perm": [], "linear_perm": [], "random": []}
                k_frac = 0.2

            for b in range(n_repeats):
                metrics = metrics_list[b]
                d = int(batch.d_emb[b].item())

                row = {
                    "dataset": name,
                    "openml_id": openml_id,
                    "repeat": b,
                    "n_train": int(batch.n_train[b].item()),
                    "n_test": int(batch.n_test[b].item()),
                    "d": d,
                }

                if batch.n_classes is not None:
                    row["n_classes"] = int(batch.n_classes[b].item())

                if evaluate_importance:
                    pfn_imp = pfn_imp_list[b]
                    ref_mi = batch.reference_importance_mi[b, :d].detach().cpu().numpy()
                    ref_rf = batch.reference_importance_rf[b, :d].detach().cpu().numpy()
                    ref_linear_perm = batch.reference_importance_linear_perm[b, :d].detach().cpu().numpy()

                    ref_mi_list.append(ref_mi)
                    ref_rf_list.append(ref_rf)
                    ref_linear_perm_list.append(ref_linear_perm)

                    rho_mi = safe_spearman(pfn_imp, ref_mi)
                    rho_rf = safe_spearman(pfn_imp, ref_rf)
                    rho_linear_perm = safe_spearman(pfn_imp, ref_linear_perm)

                    row["imp_spearman_mi"] = rho_mi
                    row["imp_spearman_rf"] = rho_rf
                    row["imp_spearman_linear_perm"] = rho_linear_perm

                    feature_perm = batch.feature_perm[b, :d].detach().cpu().numpy()
                    all_features_original = np.arange(d)

                    imp_sources = {
                        "pfn_imp": pfn_imp,
                        "mi": ref_mi,
                        "rf_perm": ref_rf,
                        "linear_perm": ref_linear_perm,
                    }

                    for method, imp in imp_sources.items():
                        topk_shuffled = topk_indices(imp, k_frac=k_frac)
                        topk_original = feature_perm[topk_shuffled]
                        selected_by_method[method].append(topk_original)

                        keep_after_remove = np.setdiff1d(all_features_original, topk_original)
                        removed_by_method[method].append(keep_after_remove)

                    rng = np.random.default_rng(b)
                    k = max(1, int(np.ceil(k_frac * d)))
                    random_original = rng.choice(d, size=k, replace=False)
                    selected_by_method["random"].append(random_original)
                    removed_by_method["random"].append(np.setdiff1d(all_features_original, random_original))

                    full_row = {
                        "dataset": name,
                        "openml_id": openml_id,
                        "repeat": b,
                        "method": "full",
                        "mode": "full",
                        "k_frac": 1.0,
                        "n_selected": d,
                        "n_train": int(batch.n_train[b].item()),
                        "n_test": int(batch.n_test[b].item()),
                        "d_selected": d,
                    }
                    full_row.update(metrics)
                    all_intervention_rows.append(full_row)

                    for j in range(d):
                        original_j = int(feature_perm[j])
                        all_imp_rows.append({
                            "dataset": name,
                            "openml_id": openml_id,
                            "repeat": b,
                            "feature_index_original": original_j,
                            "pfn_imp": float(pfn_imp[j]),
                            "ref_mi": float(ref_mi[j]),
                            "ref_rf": float(ref_rf[j]),
                            "ref_linear_perm": float(ref_linear_perm[j]),
                        })

                row.update(metrics)
                rows.append(row)
                all_metric_rows.append(row)

                metric_str = " | ".join(f"{key} {value:.4f}" for key, value in metrics.items())
                if evaluate_importance:
                    print(f"repeat {b:02d} | {metric_str} | rho_mi {rho_mi:.4f} | rho_rf {rho_rf:.4f} | rho_linear_perm {rho_linear_perm:.4f}")
                else:
                    print(f"repeat {b:02d} | {metric_str}")

            if evaluate_importance:
                for method in selected_by_method:
                    all_intervention_rows.extend(
                        evaluate_intervention(
                            model=model,
                            name=name,
                            openml_id=openml_id,
                            selected_features=selected_by_method[method],
                            method=method,
                            mode="keep",
                            k_frac=k_frac,
                            task_kind=task_kind,
                            n_repeats=n_repeats,
                        )
                    )

                    if all(len(x) >= 1 for x in removed_by_method[method]):
                        all_intervention_rows.extend(
                            evaluate_intervention(
                                model=model,
                                name=name,
                                openml_id=openml_id,
                                selected_features=removed_by_method[method],
                                method=method,
                                mode="remove",
                                k_frac=k_frac,
                                task_kind=task_kind,
                                n_repeats=n_repeats,
                            )
                        )

                print_dataset_keep_remove_diff_tables(all_intervention_rows, name)

            summary_row = {
                "dataset": name,
                "openml_id": openml_id,
                "n_repeats": n_repeats,
                "n_train": rows[0]["n_train"],
                "n_test": rows[0]["n_test"],
                "d": rows[0]["d"],
            }

            if "n_classes" in rows[0]:
                summary_row["n_classes"] = rows[0]["n_classes"]

            metadata_cols = {"dataset", "openml_id", "repeat", "n_classes", "n_train", "n_test", "d"}
            summary_keys = [key for key in rows[0].keys() if key not in metadata_cols]

            for key in summary_keys:
                mean, std = summarize([r[key] for r in rows])
                summary_row[f"{key}_mean"] = mean
                summary_row[f"{key}_std"] = std
                print(f"{key}: {mean:.4f} ± {std:.4f}")

            all_summary_rows.append(summary_row)
            print()

            del out, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(all_metric_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    run_name = f"real_eval_{n_repeats}repeats"
    save_dir = Path(__file__).resolve().parents[2] / "results" / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = save_dir / f"{run_name}_metrics_each_repeat.csv"
    summary_path = save_dir / f"{run_name}_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"saved {metrics_path}")
    print(f"saved {summary_path}")

    if evaluate_importance:
        imp_df = pd.DataFrame(all_imp_rows)
        intervention_df = pd.DataFrame(all_intervention_rows)
        importance_path = save_dir / f"{run_name}_importance_each_repeat.csv"
        intervention_path = save_dir / f"{run_name}_topk_intervention.csv"
        imp_df.to_csv(importance_path, index=False)
        intervention_df.to_csv(intervention_path, index=False)
        print(f"saved {importance_path}")
        print(f"saved {intervention_path}")


if __name__ == "__main__":
    model = TabularPFNModel(
        k=64,
        m=120,
        n_heads=4,
        depth=16,
        max_cardinality=10,
        task_kind="classification",
        max_classes=4,
    )

    model_path = "/content/best_ckpt.pt"
    main(model=model, model_path=model_path, task_kind="regression", datasets=REG_DATASETS, evaluate_importance=False)