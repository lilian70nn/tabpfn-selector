import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_X(X_train, X_test, feature_type, cardinality):
    train_parts = []
    test_parts = []

    for j in range(X_train.shape[1]):
        train_col = X_train[:, j]
        test_col = X_test[:, j]

        if int(feature_type[j]) == 0:
            valid = torch.isfinite(train_col)
            mean = train_col[valid].mean() if valid.any() else torch.tensor(0.0, device=X_train.device)
            train_filled = torch.where(torch.isfinite(train_col), train_col, mean)
            test_filled = torch.where(torch.isfinite(test_col), test_col, mean)
            std = train_filled.std(unbiased=False).clamp_min(1e-6)
            train_parts.append(((train_filled - mean) / std).unsqueeze(1))
            test_parts.append(((test_filled - mean) / std).unsqueeze(1))
        else:
            k = int(cardinality[j])
            train_missing = ~torch.isfinite(train_col)
            test_missing = ~torch.isfinite(test_col)
            train_index = torch.where(train_missing, torch.full_like(train_col, k), train_col).long().clamp(0, k)
            test_index = torch.where(test_missing, torch.full_like(test_col, k), test_col).long().clamp(0, k)
            train_parts.append(F.one_hot(train_index, num_classes=k + 1).float())
            test_parts.append(F.one_hot(test_index, num_classes=k + 1).float())

    return torch.cat(train_parts, dim=1), torch.cat(test_parts, dim=1)


def _dummy_score(y_train, y_test, classification):
    if classification:
        num_classes = int(torch.max(torch.cat((y_train, y_test))).item()) + 1
        counts = torch.bincount(y_train.long(), minlength=num_classes)
        majority = int(counts.argmax().item())
        prediction = torch.full_like(y_test.long(), majority)
        return float((prediction == y_test.long()).float().mean().item())

    prediction = torch.full_like(y_test.float(), y_train.float().mean())
    mse = F.mse_loss(prediction, y_test.float())
    variance = y_test.float().var(unbiased=False).clamp_min(1e-8)
    return float((1.0 - mse / variance).item())


class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _fit_probe(model, X_train, y_train, classification, epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = F.cross_entropy(output, y_train.long()) if classification else F.mse_loss(output[:, 0], y_train.float())

        if not torch.isfinite(loss):
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()


@torch.no_grad()
def _score_probe(model, X_test, y_test, classification):
    output = model(X_test)

    if classification:
        prediction = output.argmax(dim=1)
        return float((prediction == y_test.long()).float().mean().item())

    prediction = output[:, 0]
    mse = F.mse_loss(prediction, y_test.float())
    variance = y_test.float().var(unbiased=False).clamp_min(1e-8)
    return float((1.0 - mse / variance).item())

@torch.no_grad()
def _score_probe(model, X_test, y_test, classification):
    output = model(X_test)

    if classification:
        prediction = output.argmax(dim=1)
        return float((prediction == y_test.long()).float().mean().item())

    prediction = output[:, 0]
    mse = F.mse_loss(prediction, y_test.float())
    variance = y_test.float().var(unbiased=False).clamp_min(1e-8)
    r2 = 1.0 - mse / variance

    if r2.item() < -1.0:
        print(
            f"[BAD MLP] r2={r2.item():.4f}, "
            f"mse={mse.item():.6f}, "
            f"y_test_var={variance.item():.6f}, "
            f"y_test_std={y_test.float().std(unbiased=False).item():.6f}"
        )

    return float(r2.item())

def _linear_regression_score(X_train, y_train, X_test, y_test):
    X_train = X_train.double()
    X_test = X_test.double()
    y_train = y_train.double()
    y_test = y_test.double()

    ones_train = torch.ones((X_train.shape[0], 1), device=X_train.device, dtype=X_train.dtype)
    ones_test = torch.ones((X_test.shape[0], 1), device=X_test.device, dtype=X_test.dtype)
    X_train_aug = torch.cat((X_train, ones_train), dim=1)
    X_test_aug = torch.cat((X_test, ones_test), dim=1)

    solution = torch.linalg.lstsq(X_train_aug, y_train.unsqueeze(1)).solution
    prediction = (X_test_aug @ solution)[:, 0]

    mse = F.mse_loss(prediction, y_test)
    variance = y_test.var(unbiased=False).clamp_min(1e-12)
    r2 = 1.0 - mse / variance

    if r2.item() < -1.0:
        print(
            f"[BAD LINEAR] r2={r2.item():.4f}, "
            f"mse={mse.item():.6f}, "
            f"y_train_std={y_train.std(unbiased=False).item():.6f}, "
            f"y_test_std={y_test.std(unbiased=False).item():.6f}, "
            f"y_test_var={variance.item():.6f}"
        )

    return float(r2.item())


def compute_difficulty_metrics(X_train, X_test, y_train, y_test, num_classes, epochs, probe_seed, feature_type, cardinality):
    classification = num_classes is not None

    if not classification:
        y_all = torch.cat((y_train.float(), y_test.float()))
        q = torch.tensor([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99], device=y_all.device)

        y_all_std = y_all.std(unbiased=False).item()
        y_train_std = y_train.float().std(unbiased=False).item()
        y_test_std = y_test.float().std(unbiased=False).item()

        if y_test_std < 0.4:
            print(
                f"[LOW TEST VAR] "
                f"all_std={y_all_std:.4f}, "
                f"train_std={y_train_std:.4f}, "
                f"test_std={y_test_std:.4f}, "
                f"quantiles={torch.quantile(y_all, q).tolist()}"
            )

    X_train, X_test = _prepare_X(X_train, X_test, feature_type, cardinality)
    dummy_score = _dummy_score(y_train, y_test, classification)

    devices = [X_train.device.index] if X_train.is_cuda and X_train.device.index is not None else []

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(probe_seed))

        if classification:
            linear_probe = LinearProbe(X_train.shape[1], int(num_classes)).to(X_train.device)
            _fit_probe(linear_probe, X_train, y_train, True, epochs=epochs)
            linear_score = _score_probe(linear_probe, X_test, y_test, True)
        else:
            linear_score = _linear_regression_score(X_train, y_train, X_test, y_test)

        output_dim = int(num_classes) if classification else 1
        mlp_probe = MLPProbe(X_train.shape[1], output_dim).to(X_train.device)
        _fit_probe(mlp_probe, X_train, y_train, classification, epochs=epochs)
        mlp_score = _score_probe(mlp_probe, X_test, y_test, classification)

    return {
        "dummy_score": dummy_score,
        "linear_score": linear_score,
        "mlp_score": mlp_score,
        "signal_gain": mlp_score - dummy_score,
        "linear_gain": linear_score - dummy_score,
        "nonlinear_gain": mlp_score - linear_score,
    }