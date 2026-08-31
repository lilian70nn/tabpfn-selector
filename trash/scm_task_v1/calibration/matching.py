import math
import torch


def _valid_pair(x, y):
    x = x.detach().cpu()
    y = y.detach().cpu()
    mask = torch.isfinite(x.float()) & torch.isfinite(y.float())
    return x[mask], y[mask]


def _abs_pearson(x, y):
    x, y = _valid_pair(x, y)
    if x.numel() < 2:
        return 0.0

    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = x_centered.std(unbiased=False)
    y_std = y_centered.std(unbiased=False)

    if x_std <= 1e-8 or y_std <= 1e-8:
        return 0.0

    correlation = (x_centered * y_centered).mean() / (x_std * y_std)
    return float(correlation.abs().clamp(0.0, 1.0).item())


def _correlation_ratio(continuous, categorical):
    continuous, categorical = _valid_pair(continuous, categorical)
    if continuous.numel() < 2:
        return 0.0

    continuous = continuous.float()
    categorical = categorical.long()
    total_var = continuous.var(unbiased=False)

    if total_var <= 1e-8:
        return 0.0

    global_mean = continuous.mean()
    between_var = torch.zeros((), dtype=continuous.dtype)

    for category in torch.unique(categorical):
        mask = categorical == category
        weight = mask.float().mean()
        between_var += weight * (continuous[mask].mean() - global_mean).square()

    eta_squared = between_var / total_var.clamp_min(1e-8)
    return float(torch.sqrt(eta_squared.clamp(0.0, 1.0)).item())


def _cramers_v(x, y):
    x, y = _valid_pair(x, y)
    if x.numel() == 0:
        return 0.0

    x = x.long()
    y = y.long()
    _, x_inverse = torch.unique(x, return_inverse=True)
    _, y_inverse = torch.unique(y, return_inverse=True)
    num_x = int(x_inverse.max().item()) + 1
    num_y = int(y_inverse.max().item()) + 1

    if num_x < 2 or num_y < 2:
        return 0.0

    table = torch.zeros((num_x, num_y), dtype=torch.float32)
    table.index_put_((x_inverse, y_inverse), torch.ones_like(x_inverse, dtype=torch.float32), accumulate=True)

    n = table.sum()
    if n <= 0:
        return 0.0

    row_sum = table.sum(dim=1, keepdim=True)
    col_sum = table.sum(dim=0, keepdim=True)
    expected = row_sum @ col_sum / n
    valid = expected > 0
    chi_squared = ((table[valid] - expected[valid]).square() / expected[valid]).sum()
    denominator = n * min(num_x - 1, num_y - 1)

    if denominator <= 0:
        return 0.0

    return float(torch.sqrt(chi_squared / denominator).clamp(0.0, 1.0).item())


def _feature_pair_dependence(x, y, x_categorical, y_categorical):
    if not x_categorical and not y_categorical:
        return _abs_pearson(x, y)
    if x_categorical and y_categorical:
        return _cramers_v(x, y)
    if x_categorical:
        return _correlation_ratio(y, x)
    return _correlation_ratio(x, y)


def _safe_abs_correlation(X, feature_type=None):
    X = X.detach().float().cpu()

    if X.shape[1] < 2:
        return 0.0

    if feature_type is None:
        feature_type = torch.zeros(X.shape[1], dtype=torch.long)
    else:
        feature_type = torch.as_tensor(feature_type).long().cpu()

    if feature_type.numel() != X.shape[1]:
        raise ValueError(f"feature_type has length {feature_type.numel()}, but X has {X.shape[1]} features.")

    scores = []

    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            score = _feature_pair_dependence(
                X[:, i],
                X[:, j],
                feature_type[i].item() == 1,
                feature_type[j].item() == 1,
            )
            scores.append(score)

    return float(torch.tensor(scores).mean().item()) if scores else 0.0


def _feature_target_dependence(X, y, classification, feature_type=None):
    X = X.detach().float().cpu()
    y = y.detach().cpu()

    if feature_type is None:
        feature_type = torch.zeros(X.shape[1], dtype=torch.long)
    else:
        feature_type = torch.as_tensor(feature_type).long().cpu()

    if feature_type.numel() != X.shape[1]:
        raise ValueError(f"feature_type has length {feature_type.numel()}, but X has {X.shape[1]} features.")

    scores = []

    for j in range(X.shape[1]):
        feature = X[:, j]
        feature_is_categorical = feature_type[j].item() == 1

        if classification:
            if feature_is_categorical:
                score = _cramers_v(feature, y)
            else:
                score = _correlation_ratio(feature, y)
        else:
            if feature_is_categorical:
                score = _correlation_ratio(y, feature)
            else:
                score = _abs_pearson(feature, y)

        scores.append(score)

    return float(torch.tensor(scores).mean().item()) if scores else 0.0


def compute_observable_profile(X, y, feature_type=None, cardinality=None, num_classes=None):
    X = X.detach().float().cpu()
    y = y.detach().cpu()
    classification = num_classes is not None

    if feature_type is not None:
        feature_type = torch.as_tensor(feature_type).long().cpu()
        if feature_type.numel() != X.shape[1]:
            raise ValueError(f"feature_type has length {feature_type.numel()}, but X has {X.shape[1]} features.")

    profile = {
        "log_n": math.log(max(1, X.shape[0])),
        "log_d": math.log(max(1, X.shape[1])),
        "log_n_over_d": math.log(max(1e-6, X.shape[0] / X.shape[1])),
        "missing_rate": float(torch.isnan(X).float().mean().item()),
        "mean_abs_feature_correlation": _safe_abs_correlation(X, feature_type),
        "mean_feature_target_dependence": _feature_target_dependence(X, y, classification, feature_type),
    }

    if feature_type is not None:
        profile["categorical_fraction"] = float((feature_type == 1).float().mean().item())

    if cardinality is not None:
        cardinality = torch.as_tensor(cardinality).float().cpu()
        positive = cardinality[cardinality > 0]
        profile["mean_categorical_cardinality"] = float(positive.mean().item()) if positive.numel() else 0.0

    if classification:
        counts = torch.bincount(y.long(), minlength=int(num_classes)).float()
        probs = counts / counts.sum().clamp_min(1)
        profile["class_min_fraction"] = float(probs.min().item())
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        profile["class_entropy"] = float((entropy / math.log(max(2, int(num_classes)))).item())

    return profile


def profile_distance_to_reference(synthetic_profile, real_reference, scales=None):
    shared_metrics = sorted(set(synthetic_profile) & set(real_reference))

    if not shared_metrics:
        raise ValueError("No shared metrics between synthetic and real profiles.")

    squared_distances = []

    for metric_name in shared_metrics:
        scale = 1.0 if scales is None else float(scales.get(metric_name, 1.0))
        scale = max(scale, 1e-8)
        difference = (float(synthetic_profile[metric_name]) - float(real_reference[metric_name])) / scale
        squared_distances.append(difference * difference)

    return float(math.sqrt(sum(squared_distances) / len(squared_distances)))