from .helper import move_batch_to_device, infer_loader_use_selector
from .eval import evaluate_synthetic
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
    save_path=None,
):
    model.to(device)
    model.train()

    if save_path is not None:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write("")

    def log_line(s):
        print(s, flush=True)
        if save_path is not None:
            with open(save_path, "a") as f:
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

    train_iter = iter(train_loader)

    running_loss = 0.0
    running_pred = 0.0
    running_imp = 0.0
    running_n = 0
    running_imp_n = 0

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
        loss_dict = model.total_loss(
            batch,
            out,
            importance_weight=importance_weight,
        )

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

            if loader_use_selector:
                log_line(
                    f"[val] step {step:06d} | "
                    f"loss {val_metrics['loss']:.4f} | "
                    f"pred {val_metrics['pred_loss']:.4f} | "
                    f"imp {val_metrics['importance_loss']:.6f} | "
                    f"acc {val_metrics.get('acc', float('nan')):.4f} | "
                    f"bal_acc {val_metrics.get('balanced_acc', float('nan')):.4f} | "
                    f"f1 {val_metrics.get('macro_f1', float('nan')):.4f} | "
                    f"auc {val_metrics.get('roc_auc', float('nan')):.4f} | "
                    f"imp_corr {val_metrics.get('importance_pearson', float('nan')):.4f}"
                )
            else:
                log_line(
                    f"[val] step {step:06d} | "
                    f"loss {val_metrics['loss']:.4f} | "
                    f"pred {val_metrics['pred_loss']:.4f} | "
                    f"acc {val_metrics.get('acc', float('nan')):.4f} | "
                    f"bal_acc {val_metrics.get('balanced_acc', float('nan')):.4f} | "
                    f"f1 {val_metrics.get('macro_f1', float('nan')):.4f} | "
                    f"auc {val_metrics.get('roc_auc', float('nan')):.4f}"
                )

            model.train()