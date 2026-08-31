from pathlib import Path

from src.training.helper import move_batch_to_device, infer_loader_use_selector
from src.training.eval import evaluate_synthetic
import torch


def train_synthetic(
    model,
    train_loader,
    optimizer,
    device,
    steps=5000,
    importance_weight: float | None = None,
    grad_clip=1.0,
    log_every=50,
    val_loader=None,
    val_every=500,
    val_batches=50,
    imp_trace=True,
    trace_num_tables=10,
    save_path=None,
):
    model.to(device)
    model.train()

    save_dir = None
    log_path = None
    trace_path = None
    best_ckpt_path = None

    if save_path is not None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "train_log.txt"
        trace_path = save_dir / "importance_trace.txt"
        best_ckpt_path = save_dir / "best_ckpt.pt"
        with open(log_path, "w") as f:
            f.write("")
        if imp_trace:
            with open(trace_path, "w") as f:
                f.write("")

    def log_line(s):
        print(s, flush=True)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(s + "\n")

    def trace_line(s):
        if trace_path is not None:
            with open(trace_path, "a") as f:
                f.write(s + "\n")

    loader_use_selector = infer_loader_use_selector(train_loader)

    if loader_use_selector:
        assert importance_weight is not None
        assert importance_weight > 0
    else:
        assert importance_weight is None

    if val_loader is not None:
        val_use_selector = infer_loader_use_selector(val_loader)
        assert val_use_selector == loader_use_selector, (
            "train_loader and val_loader must use the same use_selector setting"
        )
    else:
        imp_trace = False

    train_iter = iter(train_loader)
    trace_batch = None
    actual_trace_num_tables = 0

    if imp_trace:
        if not loader_use_selector:
            raise ValueError("imp_trace=True requires use_selector=True")
        raw_trace_batch = next(iter(val_loader))
        original_batch_size = int(raw_trace_batch.X_train.shape[0])
        actual_trace_num_tables = min(int(trace_num_tables), original_batch_size)

        for name, value in vars(raw_trace_batch).items():
            if (torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_batch_size):
                setattr(raw_trace_batch, name, value[:actual_trace_num_tables].clone())

        trace_batch = move_batch_to_device(raw_trace_batch, device)

        d_list = trace_batch.d_emb.detach().cpu().tolist()
        gt_imp_list = trace_batch.feature_importance.detach().float().cpu().tolist()

        trace_line(f"[setup] num_tables={actual_trace_num_tables}")

        for table_idx, d in enumerate(d_list):
            gt_imp = gt_imp_list[table_idx][:d]
            gt_str = ",".join(f"{x:.8f}" for x in gt_imp)
            trace_line(f"[gt_imp] table={table_idx} d={d} values={gt_str}")

        model.eval()
        with torch.no_grad():
            trace_out = model(trace_batch, return_selector_layers=True)
            trace_logits_layers = trace_out["importance_logits_layers"]
            trace_scores_layers = torch.sigmoid(trace_logits_layers)

        imp_score_layers_list = trace_scores_layers.detach().float().cpu().tolist()

        for table_idx, d in enumerate(d_list):
            for layer_idx, layer_scores in enumerate(imp_score_layers_list[table_idx]):
                imp_scores = layer_scores[:d]
                score_str = ",".join(f"{x:.8f}" for x in imp_scores)
                trace_line(f"[imp_score] step=0 table={table_idx} layer={layer_idx} d={d} values={score_str}")

        model.train()


    running_loss = 0.0
    running_pred = 0.0
    running_imp = 0.0
    running_n = 0
    running_imp_n = 0
    best_pred_loss = float("inf")

    for step in range(1, steps + 1):

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        assert bool(batch.use_selector) == loader_use_selector

        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)

        loss_dict = model.total_loss(batch, out, importance_weight=importance_weight)
        loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        running_loss += float(loss_dict["loss"].detach())
        running_pred += float(loss_dict["pred_loss"].detach())
        running_n += 1

        if loader_use_selector:
            running_imp += float(loss_dict["importance_loss"].detach())
            running_imp_n += 1

        if step % log_every == 0:
            if loader_use_selector:
                log_line(
                    f"step {step:06d} | "
                    f"loss {running_loss / running_n:.4f} | "
                    f"pred {running_pred / running_n:.4f} | "
                    f"imp {running_imp / max(running_imp_n, 1):.6f}"
                )
            else:
                log_line(
                    f"step {step:06d} | "
                    f"loss {running_loss / running_n:.4f} | "
                    f"pred {running_pred / running_n:.4f}"
                )

            running_loss = 0.0
            running_pred = 0.0
            running_imp = 0.0
            running_n = 0
            running_imp_n = 0

        if val_loader is not None and step % val_every == 0:
            val_metrics = evaluate_synthetic(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=val_batches,
                importance_weight=importance_weight,
            )

            if val_metrics["pred_loss"] < best_pred_loss:
                best_pred_loss = val_metrics["pred_loss"]
                if best_ckpt_path is not None:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                            "best_pred_loss": best_pred_loss,
                            "val_metrics": val_metrics,
                        },
                        best_ckpt_path
                    ) 
                log_line(f"[best] step {step:06d} | val_pred_loss {best_pred_loss:.6f}")

            log_line(f"[val] step {step:06d} | " + " | ".join(f"{key} {value:.4f}" for key, value in val_metrics.items()))

            if imp_trace:
                model.eval()
                with torch.no_grad():
                    trace_out = model(trace_batch, return_selector_layers=True)
                    trace_logits_layers = trace_out["importance_logits_layers"]
                    trace_scores_layers = torch.sigmoid(trace_logits_layers)

                imp_score_layers_list = trace_scores_layers.detach().float().cpu().tolist()

                for table_idx, d in enumerate(d_list):
                    for layer_idx, layer_scores in enumerate(imp_score_layers_list[table_idx]):
                        imp_scores = layer_scores[:d]
                        score_str = ",".join(f"{x:.8f}" for x in imp_scores)
                        trace_line(f"[imp_score] step={step} table={table_idx} layer={layer_idx} d={d} values={score_str}")

            model.train()

