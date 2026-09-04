import torch
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from src.model.tabpfn import TabularPFNModel
from src.data.collate_real_data import collate_openml_task
from src.training.metrics import classification_metrics, regression_metrics
from test.config import CLS_DATASETS, REG_DATASETS


def evaluate_batch(model, batch, out, task_kind):

    if task_kind == "classification":
        metrics = classification_metrics(batch, out)
    elif task_kind == "regression":
        metrics = regression_metrics(batch, out, model.encoder.regression_borders)
    else:
        raise ValueError(f"Unknown task_kind: {task_kind}")

    metrics["loss"] = model.prediction_loss(batch, out).item()
    return metrics


def add_full_intervention_row(rows, name, openml_id, seed, batch, metrics):
    row = {
        "dataset": name,
        "openml_id": openml_id,
        "seed": seed,
        "method": "full",
        "mode": "full",
        "k_frac": 1.0,
        "n_selected": int(batch.d_emb.item()),
        "n_train": int(batch.n_train.item()),
        "n_test": int(batch.n_test.item()),
        "d_selected": int(batch.d_emb.item()),
    }

    row.update(metrics)
    rows.append(row)



def get_pfn_importance(batch, out):
    d = int(batch.d_emb.item())

    imp_shuffled = torch.sigmoid(out["importance_logits"][0, :d])
    imp_shuffled = imp_shuffled / (imp_shuffled.sum() + 1e-12)

    pfn_imp = torch.empty_like(imp_shuffled)
    pfn_imp[batch.feature_perm[0].to(imp_shuffled.device)] = imp_shuffled

    return pfn_imp.detach().cpu().numpy()


def eval_feature_subset(
    model,
    name,
    openml_id,
    selected_features,
    seed,
    method,
    mode,
    k_frac,
    task_kind,
):
    batch = collate_openml_task(
        [(name, openml_id)],
        use_selector=True,
        classification=(task_kind == "classification"),
        feature_seed=seed,
        shuffle_features=False,
        compute_reference_importance=False,
        reference_seed=0,
        selected_features=selected_features,
    )

    out = model(batch)
    metrics = evaluate_batch(model, batch, out, task_kind)
    row = {
        "dataset": name,
        "openml_id": openml_id,
        "seed": seed,
        "method": method,
        "mode": mode,
        "k_frac": float(k_frac),
        "n_selected": int(len(selected_features)),
        "n_train": int(batch.n_train.item()),
        "n_test": int(batch.n_test.item()),
        "d_selected": int(batch.d_emb.item()),
    }

    row.update(metrics)
    return row



def print_dataset_keep_remove_diff_tables(all_intervention_rows, name):

    df = pd.DataFrame([r for r in all_intervention_rows if r["dataset"] == name])
    metadata_cols = {
        "dataset", "openml_id", "seed", "method", "mode", "k_frac",
        "n_selected", "n_train", "n_test", "d_selected",
    }
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    full = df[df["mode"] == "full"][["seed"] + metric_cols].rename(
        columns={m: f"full_{m}" for m in metric_cols}
    )
    sub = df[df["mode"] != "full"].merge(full, on="seed", how="left")
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


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")

    rho = spearmanr(a, b).correlation
    return float(rho) if not np.isnan(rho) else float("nan")


def summarize(values):
    values = np.asarray(values, dtype=float)
    return np.nanmean(values), np.nanstd(values)

def topk_indices(imp, k_frac=0.2):
    imp = np.asarray(imp, dtype=float)
    d = len(imp)
    k = max(1, int(np.ceil(k_frac * d)))
    return np.argsort(-imp)[:k]




def main(model, model_path ,task_kind="classification", datasets=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = model_path
    ckpt = torch.load(ckpt_path, map_location=device)

    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_seeds = 10

    all_metric_rows = []
    all_imp_rows = []
    all_summary_rows = []
    all_intervention_rows = []

    with torch.no_grad():
        for name, openml_id in datasets.items():
            print("=" * 80)
            print(f"dataset: {name} ({openml_id})")

            rows = []
            pfn_imp_list = []
            ref_mi_list = []
            ref_rf_list = []
            ref_linear_perm_list = []

            for seed in range(num_seeds):
                batch = collate_openml_task(
                    [(name, openml_id)],
                    use_selector=True,
                    classification=(task_kind == "classification"),
                    feature_seed=seed,
                    shuffle_features=True,
                    compute_reference_importance=True,
                    reference_seed=0,
                )
                out = model(batch)

                metrics = evaluate_batch(model, batch, out, task_kind)

                add_full_intervention_row(
                    rows=all_intervention_rows,
                    name=name,
                    openml_id=openml_id,
                    seed=seed,
                    batch=batch,
                    metrics=metrics,
                )
                # importance: sigmoid output, normalized, restored to original feature order
                pfn_imp = get_pfn_importance(batch, out)

                d = int(batch.d_emb.item())

                ref_mi = batch.reference_importance_mi[0, :d].detach().cpu().numpy().reshape(-1)
                ref_rf = batch.reference_importance_rf[0, :d].detach().cpu().numpy().reshape(-1)
                ref_linear_perm = batch.reference_importance_linear_perm[0, :d].detach().cpu().numpy().reshape(-1)

                pfn_imp_list.append(pfn_imp)
                ref_mi_list.append(ref_mi)
                ref_rf_list.append(ref_rf)
                ref_linear_perm_list.append(ref_linear_perm)

                assert pfn_imp.shape == ref_mi.shape == ref_rf.shape == ref_linear_perm.shape, (
                    name, seed, pfn_imp.shape, ref_mi.shape, ref_rf.shape, ref_linear_perm.shape,
                )

                rho_mi = safe_spearman(pfn_imp, ref_mi)
                rho_rf = safe_spearman(pfn_imp, ref_rf)
                rho_linear_perm = safe_spearman(pfn_imp, ref_linear_perm)

                imp_sources = {
                    "pfn_imp": pfn_imp,
                    "mi": ref_mi,
                    "rf_perm": ref_rf,
                    "linear_perm": ref_linear_perm,
                }

                # ---------- top-k keep/remove intervention ----------
                k_frac = 0.2
                all_features = np.arange(d)
                for method, imp in imp_sources.items():
                    topk = topk_indices(imp, k_frac=k_frac)
                    # keep top-k
                    all_intervention_rows.append(
                        eval_feature_subset(
                            model=model,
                            name=name,
                            openml_id=openml_id,
                            selected_features=topk,
                            seed=seed,
                            method=method,
                            mode="keep",
                            k_frac=k_frac,
                            task_kind=task_kind,
                        )
                    )

                    # remove top-k
                    keep_after_remove = np.setdiff1d(all_features, topk)
                    if len(keep_after_remove) >= 1:
                        all_intervention_rows.append(
                            eval_feature_subset(
                                model=model,
                                name=name,
                                openml_id=openml_id,
                                selected_features=keep_after_remove,
                                seed=seed,
                                method=method,
                                mode="remove",
                                k_frac=k_frac,
                                task_kind=task_kind,
                            )
                        )

                # random baseline
                rng = np.random.default_rng(seed)
                k = max(1, int(np.ceil(k_frac * d)))
                random_topk = rng.choice(d, size=k, replace=False)
                all_intervention_rows.append(
                    eval_feature_subset(
                        model=model,
                        name=name,
                        openml_id=openml_id,
                        selected_features=random_topk,
                        seed=seed,
                        method="random",
                        mode="keep",
                        k_frac=k_frac,
                        task_kind=task_kind,
                    )
                )

                random_keep_after_remove = np.setdiff1d(all_features, random_topk)
                if len(random_keep_after_remove) >= 1:
                    all_intervention_rows.append(
                        eval_feature_subset(
                            model=model,
                            name=name,
                            openml_id=openml_id,
                            selected_features=random_keep_after_remove,
                            seed=seed,
                            method="random",
                            mode="remove",
                            k_frac=k_frac,
                            task_kind=task_kind,
                        )
                    )

                row = {
                    "dataset": name,
                    "openml_id": openml_id,
                    "seed": seed,
                    "n_train": int(batch.n_train.item()),
                    "n_test": int(batch.n_test.item()),
                    "d": int(batch.d_emb.item()),
                    "imp_spearman_mi": rho_mi,
                    "imp_spearman_rf": rho_rf,
                    "imp_spearman_linear_perm": rho_linear_perm,
                }

                if batch.n_classes is not None:
                    row["n_classes"] = int(batch.n_classes.item())

                row.update(metrics)

                rows.append(row)
                all_metric_rows.append(row)


                for j in range(len(pfn_imp)):
                    all_imp_rows.append({
                        "dataset": name,
                        "openml_id": openml_id,
                        "seed": seed,
                        "feature_index_original": j,
                        "pfn_imp": float(pfn_imp[j]),
                        "ref_mi": float(ref_mi[j]),
                        "ref_rf": float(ref_rf[j]),
                        "ref_linear_perm": float(ref_linear_perm[j]),

                    })

                metric_str = " | ".join(
                    f"{key} {value:.4f}" for key, value in metrics.items()
                )

                print(
                    f"seed {seed:02d} | "
                    f"{metric_str} | "
                    f"rho_mi {rho_mi:.4f} | "
                    f"rho_rf {rho_rf:.4f} | "
                    f"rho_linear_perm {rho_linear_perm:.4f}"
                )

            print()
            print_dataset_keep_remove_diff_tables(all_intervention_rows, name)

            summary_row = {
                "dataset": name,
                "openml_id": openml_id,
                "num_seeds": num_seeds,
                "n_train": rows[0]["n_train"],
                "n_test": rows[0]["n_test"],
                "d": rows[0]["d"],
            }

            if "n_classes" in rows[0]:
                summary_row["n_classes"] = rows[0]["n_classes"]

            metadata_cols = {"dataset", "openml_id", "seed", "n_classes", "n_train", "n_test", "d"}
            summary_keys = [key for key in rows[0].keys() if key not in metadata_cols]
            for key in summary_keys:
                mean, std = summarize([r[key] for r in rows])
                summary_row[f"{key}_mean"] = mean
                summary_row[f"{key}_std"] = std
                print(f"{key}: {mean:.4f} ± {std:.4f}")
            

            # importance mean/std, original feature order
            pfn_imp_arr = np.stack(pfn_imp_list, axis=0)
            ref_mi_arr = np.stack(ref_mi_list, axis=0)
            ref_rf_arr = np.stack(ref_rf_list, axis=0)
            ref_linear_perm_arr = np.stack(ref_linear_perm_list, axis=0)

            pfn_imp_mean = pfn_imp_arr.mean(axis=0)
            pfn_imp_std = pfn_imp_arr.std(axis=0)

            ref_mi_mean = ref_mi_arr.mean(axis=0)
            ref_mi_std = ref_mi_arr.std(axis=0)

            ref_rf_mean = ref_rf_arr.mean(axis=0)
            ref_rf_std = ref_rf_arr.std(axis=0)

            ref_linear_perm_mean = ref_linear_perm_arr.mean(axis=0)
            ref_linear_perm_std = ref_linear_perm_arr.std(axis=0)


            for j in range(len(pfn_imp_mean)):
                summary_row[f"pfn_imp_mean_f{j}"] = float(pfn_imp_mean[j])
                summary_row[f"pfn_imp_std_f{j}"] = float(pfn_imp_std[j])

                summary_row[f"ref_mi_mean_f{j}"] = float(ref_mi_mean[j])
                summary_row[f"ref_mi_std_f{j}"] = float(ref_mi_std[j])

                summary_row[f"ref_rf_mean_f{j}"] = float(ref_rf_mean[j])
                summary_row[f"ref_rf_std_f{j}"] = float(ref_rf_std[j])

                summary_row[f"ref_linear_perm_mean_f{j}"] = float(ref_linear_perm_mean[j])
                summary_row[f"ref_linear_perm_std_f{j}"] = float(ref_linear_perm_std[j])

            all_summary_rows.append(summary_row)

            print()

    metrics_df = pd.DataFrame(all_metric_rows)
    imp_df = pd.DataFrame(all_imp_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    intervention_df = pd.DataFrame(all_intervention_rows)

    run_name = "real_eval_10seeds"

    metrics_path = f"{run_name}_metrics_each_seed.csv"
    importance_path = f"{run_name}_importance_each_seed.csv"
    summary_path = f"{run_name}_summary.csv"
    intervention_path = f"{run_name}_topk_intervention.csv"

    metrics_df.to_csv(metrics_path, index=False)
    imp_df.to_csv(importance_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    intervention_df.to_csv(intervention_path, index=False)

    print(f"saved {metrics_path}")
    print(f"saved {importance_path}")
    print(f"saved {summary_path}")
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

    model_path = "best_ckpt-12.pt"

    main(model=model, model_path=model_path, task_kind="classification", datasets=CLS_DATASETS)