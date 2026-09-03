import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from src.model.tabpfn_v2 import TabularPFNModel
from src.data.collate_real_data import collate_openml_task



def get_probs_and_metrics(model, batch):

    out = model(batch)
    logits = out["logits"]
    test_mask = out["test_mask"]

    C = logits.shape[-1]
    class_idx = torch.arange(C, device=logits.device)[None, None, :]
    valid_class = class_idx < batch.n_classes[:, None, None]
    logits = logits.masked_fill(~valid_class, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    y_pred = logits.argmax(dim=-1)
    y_true = batch.y_test.long()

    yt = y_true[test_mask].detach().cpu().numpy()
    yp = y_pred[test_mask].detach().cpu().numpy()

    n_classes = int(batch.n_classes.item())
    probs_flat = probs[test_mask][:, :n_classes].detach().cpu().numpy()

    loss = model.prediction_loss(batch, out).item()
    acc = accuracy_score(yt, yp)
    bal_acc = balanced_accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average="macro", zero_division=0)

    try:
        if n_classes == 2:
            auc = roc_auc_score(yt, probs_flat[:, 1])
        else:
            auc = roc_auc_score(yt, probs_flat, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "out": out,
        "loss": loss,
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "auc": auc,
    }


def restore_to_original_order(values_shuffled, feature_perm):
    """
    X_shuffled = X_original[:, feature_perm]
    values_shuffled is in shuffled feature order.
    Return values in original feature order.
    """
    values_original = torch.empty_like(values_shuffled)
    values_original[feature_perm] = values_shuffled
    return values_original


def get_ours_importance_original(out, batch):
    d = int(batch.d_emb.item())

    imp_shuffled = torch.sigmoid(out["importance_logits"][0, :d])
    imp_shuffled = imp_shuffled / (imp_shuffled.sum() + 1e-12)

    imp_original = torch.empty_like(imp_shuffled)
    imp_original[batch.feature_perm[0].to(imp_shuffled.device)] = imp_shuffled

    return imp_original.detach().cpu().numpy()


def score_from_metric(metrics, metric_name="auc"):
    score = metrics[metric_name]
    if np.isnan(score):
        score = metrics["bal_acc"]
    return float(score)



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


def add_full_intervention_row(rows, name, openml_id, seed, batch, metrics):
    rows.append({
        "dataset": name,
        "openml_id": openml_id,
        "seed": seed,
        "method": "full",
        "mode": "full",
        "k_frac": 1.0,
        "n_selected": int(batch.d_emb.item()),
        "n_classes": int(batch.n_classes.item()),
        "n_train": int(batch.n_train.item()),
        "n_test": int(batch.n_test.item()),
        "d_selected": int(batch.d_emb.item()),
        "loss": metrics["loss"],
        "acc": metrics["acc"],
        "bal_acc": metrics["bal_acc"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "score": score_from_metric(metrics, "auc"),
    })


def eval_feature_subset(
    model,
    name,
    openml_id,
    selected_features,
    seed,
    method,
    mode,
    k_frac,
):
    batch = collate_openml_task(
        [(name, openml_id)],
        use_selector=True,
        classification=True,
        feature_seed=seed,
        shuffle_features=False,
        compute_reference_importance=False,
        reference_seed=0,
        selected_features=selected_features,
    )

    metrics = get_probs_and_metrics(model, batch)

    return {
        "dataset": name,
        "openml_id": openml_id,
        "seed": seed,
        "method": method,
        "mode": mode,
        "k_frac": float(k_frac),
        "n_selected": int(len(selected_features)),
        "n_classes": int(batch.n_classes.item()),
        "n_train": int(batch.n_train.item()),
        "n_test": int(batch.n_test.item()),
        "d_selected": int(batch.d_emb.item()),
        "loss": metrics["loss"],
        "acc": metrics["acc"],
        "bal_acc": metrics["bal_acc"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "score": score_from_metric(metrics, "auc"),
    }

def print_dataset_keep_remove_diff_tables(all_intervention_rows, name):
    df = pd.DataFrame([
        r for r in all_intervention_rows
        if r["dataset"] == name
    ])

    metric_cols = ["loss", "acc", "bal_acc", "f1", "auc", "score"]

    full = (
        df[df["mode"] == "full"]
        [["seed"] + metric_cols]
        .rename(columns={m: f"full_{m}" for m in metric_cols})
    )

    sub = df[df["mode"] != "full"].merge(
        full,
        on="seed",
        how="left",
    )

    def make_table(mode):
        rows = []

        for method in ["ours", "mi", "rf_perm", "logreg_perm", "random"]:
            cur = sub[(sub["method"] == method) & (sub["mode"] == mode)]

            row = {
                "method": method,
                "k_frac": float(cur["k_frac"].iloc[0]) if len(cur) else np.nan,
                "n_selected_mean": float(cur["n_selected"].mean()) if len(cur) else np.nan,
            }

            # loss: intervention loss - full loss
            # positive means worse
            row["loss_increase"] = float((cur["loss"] - cur["full_loss"]).mean()) if len(cur) else np.nan

            # higher-is-better metrics: full metric - intervention metric
            # positive means worse
            for m in ["acc", "bal_acc", "f1", "auc", "score"]:
                row[f"{m}_drop"] = float((cur[f"full_{m}"] - cur[m]).mean()) if len(cur) else np.nan

            rows.append(row)

        return pd.DataFrame(rows)

    keep_table = make_table("keep")
    remove_table = make_table("remove")

    print("\n" + "-" * 120)
    print(f"[{name}] KEEP top-k features only")
    print("Diff is relative to full features.")
    print("loss_increase = keep_loss - full_loss. Positive means worse.")
    print("metric_drop = full_metric - keep_metric. Positive means worse.")
    print("KEEP: smaller drop is better.")
    print("-" * 120)
    print(keep_table.to_string(index=False))

    print("\n" + "-" * 120)
    print(f"[{name}] REMOVE top-k features")
    print("Diff is relative to full features.")
    print("loss_increase = remove_loss - full_loss. Positive means worse.")
    print("metric_drop = full_metric - remove_metric. Positive means worse.")
    print("REMOVE: larger drop is better, because deleting important features should hurt prediction more.")
    print("-" * 120)
    print(remove_table.to_string(index=False))
    print("-" * 120)

    return keep_table, remove_table



def plot_real_eval_results(summary_df):
    # keep dataset order from evaluation
    names = summary_df["dataset"].tolist()
    x = np.arange(len(names))

    # ---------- Figure 1: mean acc with std ----------
    acc_mean = summary_df["acc_mean"].to_numpy(dtype=float)
    acc_std = summary_df["acc_std"].to_numpy(dtype=float)

    plt.figure(figsize=(max(10, 0.6 * len(names)), 5))
    plt.bar(x, acc_mean, yerr=acc_std, capsize=4)
    plt.xticks(x, names, rotation=60, ha="right")
    plt.ylabel("Mean accuracy")
    plt.xlabel("Dataset")
    plt.title("Prediction accuracy across OpenML datasets")
    plt.ylim(0.0, min(1.05, np.nanmax(acc_mean + acc_std) + 0.05))
    plt.tight_layout()
    plt.savefig("real_eval_mean_acc_std.png", dpi=300)
    plt.close()

    # ---------- Figure 2: corr between ours imp and reference imp ----------
    corr_mean = summary_df["imp_spearman_ref_mean"].to_numpy(dtype=float)
    corr_std = summary_df["imp_spearman_ref_std"].to_numpy(dtype=float)

    plt.figure(figsize=(max(10, 0.6 * len(names)), 5))
    plt.bar(x, corr_mean, yerr=corr_std, capsize=4)
    plt.xticks(x, names, rotation=60, ha="right")
    plt.ylabel("Spearman correlation")
    plt.xlabel("Dataset")
    plt.title("Importance correlation: model vs reference")
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig("real_eval_imp_corr_std.png", dpi=300)
    plt.close()


def main():

    DEFAULT_DATASETS = {
        # 2-class
        "banknote-authentication": 1462,
        "diabetes": 37,
        "breast-w": 15,
        "ionosphere": 59,
        "spambase": 44,
        "credit-g": 31,
        "kr-vs-kp": 3,
        "qsar-biodeg": 1494,
        "blood-transfusion-service-center": 1464,
        "breast-cancer": 13,

        # 3-class
        "iris": 61,
        "balance-scale": 11,
        "cmc": 45052,
        "baseball": 185,

        # 4-class
        "car": 40975,
        "car_evaluation": 43921,

    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/content/best_ckpt-3.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = TabularPFNModel(
        k=64,
        m=120,
        n_heads=4,
        depth=16,
        max_cardinality=10,
        task_kind="classification",
        max_classes=4,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_seeds = 10

    all_metric_rows = []
    all_imp_rows = []
    all_summary_rows = []
    all_intervention_rows = []


    with torch.no_grad():
        for name, openml_id in DEFAULT_DATASETS.items():
            print("=" * 80)
            print(f"dataset: {name} ({openml_id})")

            rows = []
            ours_imp_list = []
            ref_mi_list = []
            ref_rf_list = []
            ref_logreg_perm_list = []

            for seed in range(num_seeds):
                batch = collate_openml_task(
                    [(name, openml_id)],
                    use_selector=True,
                    classification=True,
                    feature_seed=seed,
                    shuffle_features=True,
                    compute_reference_importance=True,
                    reference_seed=0,
                )

                metrics = get_probs_and_metrics(model, batch)
                out = metrics["out"]

                add_full_intervention_row(
                    rows=all_intervention_rows,
                    name=name,
                    openml_id=openml_id,
                    seed=seed,
                    batch=batch,
                    metrics=metrics,
                )

                # ours importance: sigmoid output, normalized, restored to original feature order
                ours_imp = get_ours_importance_original(out, batch)

                d = int(batch.d_emb.item())

                ref_mi = batch.reference_importance_mi[0, :d].detach().cpu().numpy().reshape(-1)
                ref_rf = batch.reference_importance_rf[0, :d].detach().cpu().numpy().reshape(-1)
                ref_logreg_perm = batch.reference_importance_logreg_perm[0, :d].detach().cpu().numpy().reshape(-1)

                assert ours_imp.shape == ref_mi.shape == ref_rf.shape == ref_logreg_perm.shape, (
                    name,
                    seed,
                    ours_imp.shape,
                    ref_mi.shape,
                    ref_rf.shape,
                    ref_logreg_perm.shape,
                )

                rho_mi = safe_spearman(ours_imp, ref_mi)
                rho_rf = safe_spearman(ours_imp, ref_rf)
                rho_logreg_perm = safe_spearman(ours_imp, ref_logreg_perm)

                # ---------- top-k keep/remove intervention ----------
                k_frac = 0.2
                all_features = np.arange(d)

                imp_sources = {
                    "ours": ours_imp,
                    "mi": ref_mi,
                    "rf_perm": ref_rf,
                    "logreg_perm": ref_logreg_perm,
                }

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
                        )
                    )

                row = {
                    "dataset": name,
                    "openml_id": openml_id,
                    "seed": seed,
                    "n_classes": int(batch.n_classes.item()),
                    "n_train": int(batch.n_train.item()),
                    "n_test": int(batch.n_test.item()),
                    "d": int(batch.d_emb.item()),
                    "loss": metrics["loss"],
                    "acc": metrics["acc"],
                    "bal_acc": metrics["bal_acc"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                    "imp_spearman_mi": rho_mi,
                    "imp_spearman_rf": rho_rf,
                    "imp_spearman_logreg_perm": rho_logreg_perm,
                }

                rows.append(row)
                all_metric_rows.append(row)

                ours_imp_list.append(ours_imp)
                ref_mi_list.append(ref_mi)
                ref_rf_list.append(ref_rf)
                ref_logreg_perm_list.append(ref_logreg_perm)

                for j in range(len(ours_imp)):
                    all_imp_rows.append({
                        "dataset": name,
                        "openml_id": openml_id,
                        "seed": seed,
                        "feature_index_original": j,
                        "ours_imp": float(ours_imp[j]),
                        "ref_mi": float(ref_mi[j]),
                        "ref_rf": float(ref_rf[j]),
                        "ref_logreg_perm": float(ref_logreg_perm[j]),

                    })

                print(
                    f"seed {seed:02d} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"acc {metrics['acc']:.4f} | "
                    f"bal_acc {metrics['bal_acc']:.4f} | "
                    f"f1 {metrics['f1']:.4f} | "
                    f"auc {metrics['auc']:.4f} | "
                    f"rho_mi {rho_mi:.4f} | "
                    f"rho_rf {rho_rf:.4f} | "
                    f"rho_logreg_perm {rho_logreg_perm:.4f}"
                )

            print()

            print_dataset_keep_remove_diff_tables(
                all_intervention_rows=all_intervention_rows,
                name=name,
            )

            summary_row = {
                "dataset": name,
                "openml_id": openml_id,
                "num_seeds": num_seeds,
                "n_classes": rows[0]["n_classes"],
                "n_train": rows[0]["n_train"],
                "n_test": rows[0]["n_test"],
                "d": rows[0]["d"],
            }

            for key in [
                "loss",
                "acc",
                "bal_acc",
                "f1",
                "auc",
                "imp_spearman_mi",
                "imp_spearman_rf",
                "imp_spearman_logreg_perm",
            ]:
                mean, std = summarize([r[key] for r in rows])
                summary_row[f"{key}_mean"] = mean
                summary_row[f"{key}_std"] = std
                print(f"{key}: {mean:.4f} ± {std:.4f}")

            # importance mean/std, original feature order
            ours_imp_arr = np.stack(ours_imp_list, axis=0)
            ref_mi_arr = np.stack(ref_mi_list, axis=0)
            ref_rf_arr = np.stack(ref_rf_list, axis=0)
            ref_logreg_perm_arr = np.stack(ref_logreg_perm_list, axis=0)

            ours_imp_mean = ours_imp_arr.mean(axis=0)
            ours_imp_std = ours_imp_arr.std(axis=0)

            ref_mi_mean = ref_mi_arr.mean(axis=0)
            ref_mi_std = ref_mi_arr.std(axis=0)

            ref_rf_mean = ref_rf_arr.mean(axis=0)
            ref_rf_std = ref_rf_arr.std(axis=0)

            ref_logreg_perm_mean = ref_logreg_perm_arr.mean(axis=0)
            ref_logreg_perm_std = ref_logreg_perm_arr.std(axis=0)

            print("ours_imp_mean_original_order:", ours_imp_mean)
            print("ours_imp_std_original_order:", ours_imp_std)
            print("ref_mi_mean_original_order:", ref_mi_mean)
            print("ref_mi_std_original_order:", ref_mi_std)
            print("ref_rf_mean_original_order:", ref_rf_mean)
            print("ref_rf_std_original_order:", ref_rf_std)
            print("ref_logreg_perm_mean_original_order:", ref_logreg_perm_mean)
            print("ref_logreg_perm_std_original_order:", ref_logreg_perm_std)

            print("ours order:", np.argsort(-ours_imp))
            print("mi    order:", np.argsort(-ref_mi))
            print("rf    order:", np.argsort(-ref_rf))
            print("logreg_perm order:", np.argsort(-ref_logreg_perm))
            print("ours:", ours_imp)
            print("mi   :", ref_mi)
            print("rf   :", ref_rf)
            print("logreg_perm:", ref_logreg_perm)

            for j in range(len(ours_imp_mean)):
                summary_row[f"ours_imp_mean_f{j}"] = float(ours_imp_mean[j])
                summary_row[f"ours_imp_std_f{j}"] = float(ours_imp_std[j])

                summary_row[f"ref_mi_mean_f{j}"] = float(ref_mi_mean[j])
                summary_row[f"ref_mi_std_f{j}"] = float(ref_mi_std[j])

                summary_row[f"ref_rf_mean_f{j}"] = float(ref_rf_mean[j])
                summary_row[f"ref_rf_std_f{j}"] = float(ref_rf_std[j])

                summary_row[f"ref_logreg_perm_mean_f{j}"] = float(ref_logreg_perm_mean[j])
                summary_row[f"ref_logreg_perm_std_f{j}"] = float(ref_logreg_perm_std[j])

            all_summary_rows.append(summary_row)

            print()

    metrics_df = pd.DataFrame(all_metric_rows)
    imp_df = pd.DataFrame(all_imp_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    intervention_df = pd.DataFrame(all_intervention_rows)

    # metrics_df.to_csv("real_eval_metrics_each_seed.csv", index=False)
    # imp_df.to_csv("real_eval_importance_each_seed.csv", index=False)
    # summary_df.to_csv("real_eval_summary_10seeds.csv", index=False)
    # intervention_df.to_csv("real_eval_topk_intervention.csv", index=False)

    run_name = "real_eval_binary_multiclass_10seeds"

    metrics_path = f"{run_name}_metrics_each_seed.csv"
    importance_path = f"{run_name}_importance_each_seed.csv"
    summary_path = f"{run_name}_summary.csv"
    intervention_path = f"{run_name}_topk_intervention.csv"

    metrics_df.to_csv(metrics_path, index=False)
    imp_df.to_csv(importance_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    intervention_df.to_csv(intervention_path, index=False)

    # ---------- top-k keep/remove intervention ----------
    k_frac = 0.2
    all_features = np.arange(d)

    imp_sources = {
        "ours": ours_imp,
        "mi": ref_mi,
        "rf_perm": ref_rf,
        "logreg_perm": ref_logreg_perm,
    }

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
                )
            )

    #plot_real_eval_results(summary_df)

    print("saved real_eval_metrics_each_seed.csv")
    print("saved real_eval_importance_each_seed.csv")
    print("saved real_eval_summary_10seeds.csv")
    print("saved real_eval_topk_intervention.csv")
    # print("saved real_eval_mean_acc_std.png")
    # print("saved real_eval_imp_corr_std.png")



if __name__ == "__main__":
    main()