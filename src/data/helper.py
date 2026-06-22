import torch

def make_gen(device, seed):
    g = torch.Generator(device=device)
    if seed is None:
        seed = int(g.seed())
    else:
        seed = int(seed)
        g.manual_seed(seed)
    return g, seed



def stratified_classification_split(y, test_frac, generator, device):
    y = y.long()
    classes = torch.unique(y, sorted=True)

    train_parts = []
    test_parts = []

    for c in classes:
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(idx.numel(), device=device, generator=generator)]

        n_c = idx.numel()
        n_test_c = int(round(float(n_c) * float(test_frac)))

        if n_c >= 2:
            n_test_c = max(1, min(n_test_c, n_c - 1))
        else:
            n_test_c = 0

        test_parts.append(idx[:n_test_c])
        train_parts.append(idx[n_test_c:])

    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)

    train_idx = train_idx[
        torch.randperm(train_idx.numel(), device=device, generator=generator)
    ]
    test_idx = test_idx[
        torch.randperm(test_idx.numel(), device=device, generator=generator)
    ]

    return train_idx, test_idx


def discretize_latent_random_bins(
    latent_y,
    C,
    generator,
    min_per_class=2,
    alpha=5.0,
):
    n = latent_y.shape[0]
    device = latent_y.device

    assert C >= 2
    assert n >= C * min_per_class, (
        f"Need n >= C * min_per_class, got n={n}, C={C}, "
        f"min_per_class={min_per_class}"
    )

    order = torch.argsort(latent_y)

    weights = torch.rand(C, device=device, generator=generator)
    weights = weights.pow(1.0 / float(alpha))
    props = weights / weights.sum().clamp_min(1e-12)

    remaining_n = n - C * min_per_class
    counts = torch.floor(props * remaining_n).long()
    counts = counts + min_per_class

    diff = int(n - counts.sum().item())

    if diff > 0:
        extra_idx = torch.randperm(C, device=device, generator=generator)
        for k in range(diff):
            counts[extra_idx[k % C]] += 1

    elif diff < 0:
        need = -diff
        for c in torch.randperm(C, device=device, generator=generator).tolist():
            removable = int((counts[c] - min_per_class).item())
            take = min(removable, need)
            counts[c] -= take
            need -= take
            if need == 0:
                break
        assert need == 0

    assert int(counts.sum().item()) == n
    assert int(counts.min().item()) >= min_per_class

    y = torch.empty(n, device=device, dtype=torch.long)

    start = 0
    for c in range(C):
        end = start + int(counts[c].item())
        y[order[start:end]] = c
        start = end

    return y


def build_cell_mask(
    B,
    Ntr_max,
    Nte_max,
    d_max,
    n_train,
    n_test,
    d_emb,
    device,
    use_selector=False,
):
    if not torch.is_tensor(n_train):
        n_train = torch.tensor(n_train, device=device, dtype=torch.long)
    else:
        n_train = n_train.to(device=device, dtype=torch.long)

    if not torch.is_tensor(n_test):
        n_test = torch.tensor(n_test, device=device, dtype=torch.long)
    else:
        n_test = n_test.to(device=device, dtype=torch.long)

    if not torch.is_tensor(d_emb):
        d_emb = torch.tensor(d_emb, device=device, dtype=torch.long)
    else:
        d_emb = d_emb.to(device=device, dtype=torch.long)

    N = Ntr_max + 1 + Nte_max
    F = d_max + 1
    selector_idx = Ntr_max
    test_start = Ntr_max + 1
    y_slot = d_max

    idx_N = torch.arange(N, device=device).view(1, N)
    train_ok = idx_N < n_train.view(B, 1)
    test_ok = (idx_N >= test_start) & (idx_N < (test_start + n_test).view(B, 1))
    normal_row_ok = train_ok | test_ok
    idx_F = torch.arange(F, device=device).view(1, F)
    feat_ok = idx_F < d_emb.view(B, 1)
    y_ok = idx_F == y_slot
    normal_slot_ok = feat_ok | y_ok
    cell_mask = normal_row_ok[:, :, None] & normal_slot_ok[:, None, :]

    if use_selector:
        selector_ok = idx_N == selector_idx
        # selector row: only real feature slots, no y slot
        selector_cell_mask = selector_ok[:, :, None] & feat_ok[:, None, :]
        cell_mask = cell_mask | selector_cell_mask

    return cell_mask

