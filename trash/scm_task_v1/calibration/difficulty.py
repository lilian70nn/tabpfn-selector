import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_X(X_train, X_test, feature_type=None, cardinality=None):
    X_train = X_train.detach().float().cpu()
    X_test = X_test.detach().float().cpu()
    d = X_train.shape[1]

    if feature_type is None:
        feature_type = torch.zeros(d, dtype=torch.long)
    else:
        feature_type = torch.as_tensor(feature_type).long().cpu()

    if feature_type.numel() != d:
        raise ValueError(f"feature_type has length {feature_type.numel()}, but X has {d} features.")

    if cardinality is not None:
        cardinality = torch.as_tensor(cardinality).long().cpu()
        if cardinality.numel() != d:
            raise ValueError(f"cardinality has length {cardinality.numel()}, but X has {d} features.")

    train_parts = []
    test_parts = []

    for j in range(d):
        train_col = X_train[:, j]
        test_col = X_test[:, j]
        is_categorical = feature_type[j].item() == 1

        if not is_categorical:
            train_mean = torch.nanmean(train_col)
            if not torch.isfinite(train_mean):
                train_mean = torch.tensor(0.0)

            train_col = torch.where(torch.isnan(train_col), train_mean, train_col)
            test_col = torch.where(torch.isnan(test_col), train_mean, test_col)

            mean = train_col.mean()
            std = train_col.std(unbiased=False).clamp_min(1e-6)
            train_col = ((train_col - mean) / std).unsqueeze(1)
            test_col = ((test_col - mean) / std).unsqueeze(1)

            train_parts.append(train_col)
            test_parts.append(test_col)

        else:
            train_valid = train_col[torch.isfinite(train_col)]
            test_valid = test_col[torch.isfinite(test_col)]

            if cardinality is not None and cardinality[j].item() > 0:
                num_categories = int(cardinality[j].item())
            elif train_valid.numel() > 0:
                num_categories = int(train_valid.long().max().item()) + 1
            else:
                num_categories = 0

            # Reserve one extra category for missing / unseen values.
            unknown_index = num_categories
            output_dim = num_categories + 1

            train_codes = torch.full((train_col.shape[0],), unknown_index, dtype=torch.long)
            test_codes = torch.full((test_col.shape[0],), unknown_index, dtype=torch.long)

            train_mask = torch.isfinite(train_col)
            test_mask = torch.isfinite(test_col)

            if train_mask.any():
                codes = train_col[train_mask].long()
                valid_codes = (codes >= 0) & (codes < num_categories)
                indices = torch.where(train_mask)[0]
                train_codes[indices[valid_codes]] = codes[valid_codes]

            if test_mask.any():
                codes = test_col[test_mask].long()
                valid_codes = (codes >= 0) & (codes < num_categories)
                indices = torch.where(test_mask)[0]
                test_codes[indices[valid_codes]] = codes[valid_codes]

            train_one_hot = F.one_hot(train_codes, num_classes=output_dim).float()
            test_one_hot = F.one_hot(test_codes, num_classes=output_dim).float()

            train_parts.append(train_one_hot)
            test_parts.append(test_one_hot)

    return torch.cat(train_parts, dim=1), torch.cat(test_parts, dim=1)


def _classification_dummy(y_train, y_test):
    values, counts = torch.unique(y_train, return_counts=True)
    majority = values[counts.argmax()]
    prediction = torch.full_like(y_test, majority)
    return float((prediction == y_test).float().mean().item())


def _regression_dummy(y_train, y_test):
    prediction = torch.full_like(y_test.float(), y_train.float().mean())
    mse = F.mse_loss(prediction, y_test.float())
    variance = y_test.float().var(unbiased=False).clamp_min(1e-8)
    return float((1.0 - mse / variance).item())


class LinearProbe(nn.Module):
    def __init__(self, d, output_dim):
        super().__init__()
        self.linear = nn.Linear(d, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, d, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def _fit_probe(model, X_train, y_train, classification, epochs=100, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = F.cross_entropy(output, y_train.long()) if classification else F.mse_loss(output[:, 0], y_train.float())
        loss.backward()
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


def compute_difficulty_metrics(X_train, y_train, X_test, y_test, num_classes, epochs=1000, mlp_hidden_dim=64, probe_seed=0, feature_type=None, cardinality=None):
    X_train, X_test = _prepare_X(X_train, X_test, feature_type=feature_type, cardinality=cardinality)
    y_train = y_train.detach().cpu()
    y_test = y_test.detach().cpu()
    classification = num_classes is not None

    if classification:
        dummy_score = _classification_dummy(y_train, y_test)
        output_dim = int(num_classes)
    else:
        dummy_score = _regression_dummy(y_train, y_test)
        output_dim = 1

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(probe_seed))
        linear = LinearProbe(X_train.shape[1], output_dim)
        mlp = MLPProbe(X_train.shape[1], output_dim, hidden_dim=mlp_hidden_dim)

    _fit_probe(linear, X_train, y_train, classification, epochs=epochs)
    _fit_probe(mlp, X_train, y_train, classification, epochs=epochs)

    linear_score = _score_probe(linear, X_test, y_test, classification)
    mlp_score = _score_probe(mlp, X_test, y_test, classification)

    return {
        "dummy_score": dummy_score,
        "linear_score": linear_score,
        "mlp_score": mlp_score,
        "signal_gain": mlp_score - dummy_score,
        "linear_gain": linear_score - dummy_score,
        "nonlinear_gain": mlp_score - linear_score,
    }