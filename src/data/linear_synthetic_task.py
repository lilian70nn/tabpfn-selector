import torch
from .generation_helper import make_gen, stratified_classification_split, discretize_latent_random_bins
from .synthetic_task import GenerateTask


class MixedLinearTask(GenerateTask):
    """
    Mixed tabular synthetic prior.

    Supports:
    - continuous features
    - categorical features
    - missing values in observed X
    - regression if num_classes is None
    - classification if num_classes >= 2

    Key convention:
    - X_clean is used to generate y.
    - X_obs is shown to the model and may contain NaN.
    - categorical values are stored as category ids: 0, 1, ..., K_j - 1.
    - categorical features affect y through lookup effects, not id * weight.
    """

    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        num_classes=None,          # None = regression, int >= 2 = classification
        n_max=500,
        d_max=20,
        n_min=128,
        d_min=2,
        test_frac=0.15,
        p_categorical=0.3,
        max_cardinality=10,
        p_active=0.5,
        p_missing=0.05,
        noise_level=0.1,
        device=None,
        dag_seed=None,
        aleatoric_seed=None,
        x_seed=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device

        self.num_classes = num_classes
        self.n_max = int(n_max)
        self.d_max = int(d_max)
        self.n_min = int(n_min)
        self.d_min = int(d_min)
        self.test_frac = float(test_frac)
        self.p_categorical = float(p_categorical)
        self.max_cardinality = int(max_cardinality)
        self.p_active = float(p_active)
        self.p_missing = float(p_missing)
        self.noise_level = float(noise_level)

        if self.num_classes is not None:
            self.num_classes = int(self.num_classes)
            assert self.num_classes >= 2, "num_classes must be None or >= 2"

        assert self.d_max >= 2
        assert self.n_max >= 32
        assert self.n_min >= 3
        assert self.n_max >= self.n_min
        assert self.d_min >= 1
        assert self.d_max >= self.d_min
        assert 0.0 < self.test_frac < 1.0
        assert 0.0 <= self.p_categorical <= 1.0
        assert self.max_cardinality >= 2
        assert 0.0 <= self.p_active <= 1.0
        assert 0.0 <= self.p_missing <= 1.0
        assert self.noise_level >= 0.0

        self.g_dag, self.dag_seed = make_gen(device, dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(device, aleatoric_seed)
        self.g_x, self.x_seed = make_gen(device, x_seed)

        self.d = torch.randint(d_min, d_max + 1, (1,), device=device, generator=self.g_dag).item()
        self.n = torch.randint(n_min, n_max + 1, (1,), device=device, generator=self.g_dag).item()

        super().__init__()


    def _generate(self):

        n, d, device = self.n, self.d, self.device

        # 1. Decide feature types
        is_cat = torch.rand(d, device=device, generator=self.g_dag) < self.p_categorical
        feature_type = is_cat.long()
        cardinality = torch.zeros(d, device=device, dtype=torch.long)
        for j in range(d):
            if bool(is_cat[j]):
                cardinality[j] = torch.randint(2, self.max_cardinality + 1, (1,), device=device, generator=self.g_dag,).item()

        # 2. Generate X_clean
        X_clean = torch.empty(n, d, device=device, dtype=torch.float32)
        for j in range(d):
            if bool(is_cat[j]):
                K_j = int(cardinality[j].item())
                probs = torch.rand(K_j, device=device, generator=self.g_dag)
                probs = probs / probs.sum().clamp_min(1e-12)
                X_clean[:, j] = torch.multinomial(probs, num_samples=n, replacement=True, generator=self.g_x,).float()

            else:
                scale = torch.exp(0.5 * torch.randn((), device=device, generator=self.g_dag))
                shift = torch.randn((), device=device, generator=self.g_dag)
                X_clean[:, j] = (
                    scale * torch.randn(n, device=device, generator=self.g_x)
                    + shift
                )

        # 3. Decide active features
        active = torch.rand(d, device=device, generator=self.g_dag) < self.p_active
        if not bool(active.any()):
            idx = torch.randint(0, d, (1,), device=device, generator=self.g_dag).item()
            active[idx] = True
        feature_strength = torch.zeros(d, device=device, dtype=torch.float32)

        # 4. Generate scalar latent_y for both regression and classification
        latent_y = torch.zeros(n, device=device, dtype=torch.float32)
        for j in range(d):
            if not bool(active[j]):
                continue
            if bool(is_cat[j]):
                K_j = int(cardinality[j].item())
                # categorical feature: lookup scalar effect
                effects = torch.randn(K_j, device=device, generator=self.g_dag)
                xj = X_clean[:, j].long().clamp(0, K_j - 1)
                contrib_j = effects[xj]  # [n]
            else:
                # continuous feature: scalar linear effect
                w_j = torch.randn((), device=device, generator=self.g_dag)
                contrib_j = w_j * X_clean[:, j]  # [n]
            latent_y = latent_y + contrib_j
            feature_strength[j] = contrib_j.std(unbiased=False)

        # 5. Add noise
        noise_scale = self.noise_level * latent_y.std(unbiased=False).clamp_min(1e-6)
        noise = noise_scale * torch.randn(n, device=device, generator=self.g_aleatoric,)
        latent_y = latent_y + noise

        # 6. Convert latent_y to y
        if self.num_classes is None:
            # Regression: y is continuous
            y = latent_y
            self.n_classes = None
        else:
            # Classification: discretize latent_y into C quantile bins
            C = int(self.num_classes)
            y = discretize_latent_random_bins(
                latent_y=latent_y,
                C=C,
                generator=self.g_aleatoric,
                min_per_class=2,
                alpha=5.0,
            )
            self.n_classes = C

        # 7. Add missing values to observed X
        X_obs = X_clean.clone()
        if self.p_missing > 0:
            missing_mask = (
                torch.rand(X_obs.shape, device=device, generator=self.g_x)
                < self.p_missing
            )
            X_obs[missing_mask] = torch.nan
        else:
            missing_mask = torch.zeros_like(X_obs, dtype=torch.bool)

        # 8. Split train/test
        if self.num_classes is not None:
            train_idx, test_idx = stratified_classification_split(
                y=y.long(),
                test_frac=self.test_frac,
                generator=self.g_x,
                device=device,
            )
        else:
            n_test = max(1, int(round(self.n * self.test_frac)))
            n_test = min(n_test, self.n - 2)
            n_train = self.n - n_test
            perm = torch.randperm(self.n, device=device, generator=self.g_x)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

        X_train = X_obs[train_idx]
        y_train = y[train_idx]
        X_test = X_obs[test_idx]
        y_test = y[test_idx]

        # 9. Metadata
        self.n_features = d
        self.feature_type = feature_type
        self.cardinality = cardinality

        eps = 1e-8
        is_active = (feature_strength > eps).float()
        importance_ratio = feature_strength / feature_strength.sum().clamp_min(1e-12)

        info = {
            "feature_type": feature_type,
            "cardinality": cardinality,
            "is_active": is_active,
            "importance_ratio": importance_ratio,
            "feature_strength": feature_strength,
            "sampled_active": active.float(),
            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],
        }

        return X_train, y_train, X_test, y_test, info

    def visualize(self):
        return None

    def forward(self, X: torch.Tensor):
        return None
    

