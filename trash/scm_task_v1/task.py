import torch
from src.data.synthetic_task import GenerateTask
from ..utils import (
    make_gen,
    stratified_classification_split,
    detach_tree,
)
from .utils import rand, randint
from .scm import WeightedLayeredScalarSCM
from .observation import ScalarObservationHead
from .priors import (
    sample_uniform,
    sample_loguniform,
    sample_dirichlet,
    sample_connection_probs
)


class SCMTask(GenerateTask):
    """
    Mixed tabular SCM with one-dimensional continuous latent nodes.
    Underlying SCM: every node is continuous and scalar.
    Observed feature: continuous scalar, prototype category, or binned category.
    Target: direct scalar readout from final target node.
    """

    use_inference_mode = False

    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        num_classes=None,
        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_missing=0.05,
        sampling_penalty=0.25,
        device=None,
        dag_seed=None,
        aleatoric_seed=None,
        x_seed=None,
        

        num_roots=4,
        num_layers=5,
        hidden_width_min=8,
        hidden_width_max=12,
        final_width=1,
        connection_probs=((0.10, 0.30), (0.10, 0.35), (0.20, 0.50), (0.40, 0.80)),
        edge_weight_concentration=(0.30, 0.80),
        latent_noise_scale=(0.0, 0.0),
        child_method_probs=(3, 2, 3, 2),
        source_prior_probs=(0.45, 0.20, 0.15, 0.05),
        joint_mlp_hidden_dim=8,
        edge_family_probs=(4.0, 3.0, 2.0),
        small_mlp_hidden_dim=8,
        soft_tree_depth=2,
        soft_tree_temperature=0.5,




        observation_type_probs=(6.0, 2.0, 2.0),
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        min_samples_per_category=8,
        min_component_weight=0.05,
        observation_noise_scale=0.03,


        
    ):

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_classes = None if num_classes is None else int(num_classes)
        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError("num_classes must be None or at least 2.")

        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.test_frac = float(test_frac)
        self.p_missing = float(p_missing)
        self.sampling_penalty = float(sampling_penalty)
        self.g_dag, self.dag_seed = make_gen(self.device,dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(self.device, aleatoric_seed)
        self.g_x, self.x_seed = make_gen(self.device, x_seed)
        self.n = int(randint(self.n_min, self.n_max + 1, (), self.g_dag, self.device).item())
        self.d = int(randint(self.d_min, self.d_max + 1, (), self.g_dag, self.device,).item())


        self.num_roots = num_roots
        self.num_layers = num_layers
        self.hidden_width_min = hidden_width_min
        self.hidden_width_max = hidden_width_max
        self.final_width = final_width
        self.connection_probs = sample_connection_probs(
            connection_probs,
            generator=self.g_dag,
            device=self.device,
            expected_len=int(num_layers) - 1,
        )
        self.edge_weight_concentration = sample_loguniform(
            edge_weight_concentration,
            generator=self.g_dag,
            device=self.device,
        )
        self.latent_noise_scale = sample_uniform(
            latent_noise_scale,
            generator=self.g_dag,
            device=self.device,
        )
        self.child_method_probs = sample_dirichlet(
            child_method_probs,
            generator=self.g_dag,
            device=self.device,
            expected_len=4,
        )
        self.source_prior_probs = source_prior_probs
        self.joint_mlp_hidden_dim = joint_mlp_hidden_dim
        self.edge_family_probs = sample_dirichlet(
            edge_family_probs,
            generator=self.g_dag,
            device=self.device,
            expected_len=3,
        )
        self.small_mlp_hidden_dim = small_mlp_hidden_dim
        self.soft_tree_depth = soft_tree_depth
        self.soft_tree_temperature = soft_tree_temperature



        self.observation_type_probs = sample_dirichlet(
            observation_type_probs,
            generator=self.g_dag,
            device=self.device,
            expected_len=3,
        )
        self.categorical_cardinalities = categorical_cardinalities
        self.categorical_cardinality_probs = categorical_cardinality_probs
        self.min_samples_per_category = min_samples_per_category
        self.min_component_weight = min_component_weight
        self.observation_noise_scale = observation_noise_scale

        
        self.importance_eps = 1e-3


        self.observation_kwargs = dict(
            observation_type_probs=self.observation_type_probs,
            categorical_cardinalities=self.categorical_cardinalities,
            categorical_cardinality_probs=self.categorical_cardinality_probs,
            min_samples_per_category=self.min_samples_per_category,
            min_component_weight=self.min_component_weight,
            observation_noise_scale=self.observation_noise_scale,
        )



        
        super().__init__()

    @staticmethod
    def _flatten(all_latents):
        values = []
        index = []

        for layer_idx, layer in enumerate(all_latents):
            for node_idx, value in enumerate(layer):
                values.append(value)
                index.append((layer_idx, node_idx))
        return values, index


    def _apply_sampling_penalty(self, scores, selected_global_id, flat_index, penalty):
        updated = scores.clone()
        selected_layer, selected_node = flat_index[selected_global_id]
        index_lookup = {key: global_id for global_id, key in enumerate(flat_index)}

        ancestor_distance = {}
        frontier = {(selected_layer, selected_node): 0}
        while frontier:
            next_frontier = {}
            for (layer_idx, node_idx), distance in frontier.items():
                if layer_idx == 0:
                    continue
                connection = self.scm.connections[layer_idx - 1]
                parents = torch.where(connection.adj[:, node_idx])[0].tolist()
                for parent_idx in parents:
                    key = (layer_idx - 1, parent_idx)
                    new_distance = distance + 1
                    if key not in ancestor_distance or new_distance < ancestor_distance[key]:
                        ancestor_distance[key] = new_distance
                        next_frontier[key] = new_distance
            frontier = next_frontier

        descendant_distance = {}
        frontier = {(selected_layer, selected_node): 0}
        while frontier:
            next_frontier = {}
            for (layer_idx, node_idx), distance in frontier.items():
                if layer_idx >= len(self.scm.widths) - 2:
                    continue
                connection = self.scm.connections[layer_idx]
                children = torch.where(connection.adj[node_idx])[0].tolist()
                for child_idx in children:
                    key = (layer_idx + 1, child_idx)
                    new_distance = distance + 1
                    if key not in descendant_distance or new_distance < descendant_distance[key]:
                        descendant_distance[key] = new_distance
                        next_frontier[key] = new_distance
            frontier = next_frontier

        related_distance = {}
        for key, distance in ancestor_distance.items():
            related_distance[key] = distance
        for key, distance in descendant_distance.items():
            related_distance[key] = distance

        for key, distance in related_distance.items():
            global_id = index_lookup[key]
            factor = penalty ** (1.0 / float(distance))
            updated[global_id] *= factor

        updated[selected_global_id] = 0.0
        updated /= updated.sum().clamp_min(1e-12)
        return updated

    def _sample_feature_ids(self, flat_index, flat_influence, penalty):
        candidates = torch.tensor([global_id for global_id, (layer_idx, _) in enumerate(flat_index) if layer_idx < len(self.scm.widths) - 1], device=self.device, dtype=torch.long)
        d = min(self.d, int(candidates.numel()))
        sampling_scores = flat_influence.clone()
        candidate_mask = torch.zeros_like(sampling_scores, dtype=torch.bool)
        candidate_mask[candidates] = True
        sampling_scores[~candidate_mask] = 0.0
        selected = []
        for _ in range(d):
            available_scores = sampling_scores[candidates]
            if available_scores.sum() <= 0:
                remaining = torch.tensor([node_id for node_id in candidates.tolist() if node_id not in selected], device=self.device, dtype=torch.long)
                chosen = remaining[torch.randint(remaining.numel(), (1,), generator=self.g_dag, device=self.device)].item()
            else:
                probs = available_scores / available_scores.sum()
                position = torch.multinomial(probs, 1, generator=self.g_dag).item()
                chosen = int(candidates[position].item())
            selected.append(chosen)
            sampling_scores = self._apply_sampling_penalty(sampling_scores, chosen, flat_index, penalty=penalty)
        permutation = torch.randperm(len(selected), generator=self.g_dag, device=self.device)
        selected_tensor = torch.tensor(selected, device=self.device, dtype=torch.long)[permutation]
        return selected_tensor.tolist()


    def _observe_features(self, flat_latents, feature_ids):
        n = flat_latents[0].shape[0]
        d = len(feature_ids)
        X = torch.empty(n, d, device=self.device, dtype=torch.float32)
        feature_type = torch.empty(d, device=self.device, dtype=torch.long)
        cardinality = torch.zeros(d, device=self.device, dtype=torch.long)
        type_ids = torch.empty(d, device=self.device, dtype=torch.long)
        quality = torch.zeros(d, device=self.device, dtype=torch.float32)
        retention = torch.ones(d, device=self.device, dtype=torch.float32)
        categorical_features_ok = True

        type_names = []
        prototypes = []
        thresholds = []
        heads = []
        for column, node_id in enumerate(feature_ids):
            head = ScalarObservationHead(
                generator=self.g_dag,
                device=self.device,
                **self.observation_kwargs,
            )
            observed = head.observe(flat_latents[node_id], self.g_aleatoric)

            if observed.is_categorical:
                counts = torch.bincount(observed.values.long(), minlength=observed.cardinality)
                if bool((counts == 0).any()):
                    categorical_features_ok = False
                elif counts.min().float() / counts.max().float() < 0.05:
                    categorical_features_ok = False

            X[:, column] = observed.values.float()
            feature_type[column] = self.CATEGORICAL if observed.is_categorical else self.CONTINUOUS
            cardinality[column] = observed.cardinality
            type_ids[column] = observed.observation_type_id
            quality[column] = observed.quality_score
            retention[column] = observed.retention
            type_names.append(observed.observation_type_name)
            prototypes.append(observed.prototypes)
            thresholds.append(observed.thresholds)
            heads.append(head)

        return (X, feature_type, cardinality, type_ids, type_names, 
                quality, retention, prototypes, thresholds, heads, categorical_features_ok)


    def _generate(self):

        with torch.enable_grad():
        
            self.scm = WeightedLayeredScalarSCM(
                self.g_dag,
                self.g_x,
                self.g_aleatoric,
                self.num_roots,
                self.num_layers,
                self.hidden_width_min,
                self.hidden_width_max,
                self.final_width,
                self.connection_probs,
                self.edge_weight_concentration,
                self.latent_noise_scale,
                self.child_method_probs,
                self.source_prior_probs,
                self.joint_mlp_hidden_dim,
                self.edge_family_probs,
                self.small_mlp_hidden_dim,
                self.soft_tree_depth,
                self.soft_tree_temperature,
                self.device,
            )

            all_latents = self.scm.forward(self.n, latent_noise_scale=self.latent_noise_scale)

            flat_latents, flat_index = self._flatten(all_latents)
            layer_influence = self.scm.compute_sampling_influence(target_node_idx=0)
            flat_influence = torch.cat(layer_influence)
            flat_influence = flat_influence / flat_influence.sum().clamp_min(1e-12)


            feature_ids = self._sample_feature_ids(flat_index, flat_influence, penalty=self.sampling_penalty)
            self.d = len(feature_ids)

            selected_node_indices = [flat_index[global_id] for global_id in feature_ids]
            feature_strength = self.scm.compute_node_influence(all_latents=all_latents, 
                                                            node_indices=selected_node_indices,
                                                            target_node_idx=0
                                                            )

        (X_clean,feature_type,cardinality,type_ids, type_names,quality,
         feature_retention, prototypes, thresholds, heads, categorical_features_ok) = self._observe_features(flat_latents, feature_ids)
        feature_importance = feature_strength * feature_retention
        importance_ok = bool(feature_importance.max() >= self.importance_eps)
        feature_importance = feature_importance / feature_importance.sum().clamp_min(1e-12)
        

        target_global_id = sum(self.scm.widths[:-1])
        feature_ids_tensor = torch.tensor(feature_ids, device=self.device, dtype=torch.long)

        X_observed = X_clean.clone()
        missing_mask = rand(*X_observed.shape, generator=self.g_x, device=self.device) < self.p_missing
        X_observed[missing_mask] = torch.nan

        target_head = ScalarObservationHead(generator=self.g_dag, device=self.device, **self.observation_kwargs)
        target_latent = flat_latents[target_global_id]

        if self.num_classes is None:
            target_observed = target_head._continuous(target_latent.float(), self.g_aleatoric)
            y = target_observed.values
            self.n_classes = None
            target_ok = True
        else:
            target_observed = target_head.observe_categorical(target_latent, self.g_aleatoric, k=self.num_classes)
            y = target_observed.values.long()
            self.n_classes = self.num_classes

            counts = torch.bincount(y, minlength=self.num_classes)

            target_ok = (not bool((counts == 0).any())) and bool(counts.min().float() / counts.max().float() >= 0.05)

        is_valid = categorical_features_ok and target_ok and importance_ok

        if self.num_classes is not None:
            train_idx, test_idx = stratified_classification_split(
                y=y.long(),
                test_frac=self.test_frac,
                generator=self.g_x,
                device=self.device,   
            )
        else:
            n_test = min(max(1, round(self.n * self.test_frac)), self.n - 2)
            order = torch.randperm(self.n, generator=self.g_x, device=self.device)
            train_idx = order[:-n_test]
            test_idx = order[-n_test:]
        

        info = {
            "sampled_connection_probs": torch.tensor(self.connection_probs, device=self.device, dtype=torch.float32),
            "sampled_edge_weight_concentration": torch.tensor(self.edge_weight_concentration, device=self.device, dtype=torch.float32),
            "sampled_latent_noise_scale": torch.tensor(self.latent_noise_scale, device=self.device, dtype=torch.float32),
            "sampled_observation_type_probs": torch.tensor(self.observation_type_probs, device=self.device, dtype=torch.float32),
            "sampled_edge_family_probs": torch.tensor(self.edge_family_probs, device=self.device, dtype=torch.float32),
            "sampled_child_method_probs": torch.tensor(self.child_method_probs, device=self.device, dtype=torch.float32),
            "feature_type": feature_type,
            "cardinality": cardinality,
            "feature_observation_type_ids": type_ids,
            "feature_observation_type_names": type_names,
            "feature_observation_quality": quality,
            "feature_retention": feature_retention,
            "feature_prototypes": prototypes,
            "feature_thresholds": thresholds,
            "feature_ids": feature_ids_tensor,
            "target_id": torch.tensor(target_global_id, device=self.device, dtype=torch.long),
            "feature_importance": feature_importance,
            "is_valid": is_valid,
            "categorical_features_ok": categorical_features_ok,
            "target_ok": target_ok,
            "importance_ok": importance_ok,
            "all_node_influence": flat_influence,
            "layer_node_influence": layer_influence,
            "layer_widths": torch.tensor(self.scm.widths, device=self.device, dtype=torch.long),
            "connection_probs": torch.tensor(self.scm.connection_probs, device=self.device, dtype=torch.float32),
            "adjacency_matrices": [connection.adj for connection in self.scm.connections],
            "edge_weight_matrices": [connection.weights for connection in self.scm.connections],
            "latent_dim": torch.tensor(1, device=self.device, dtype=torch.long),
            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],
        }

        self.feature_type = feature_type
        self.cardinality = cardinality
        self.feature_observation_heads = heads
        self.target_observation_head = target_head
        self.n_features = self.d

        result = (
            X_observed[train_idx], y[train_idx],
            X_observed[test_idx], y[test_idx], info
        )

        result = detach_tree(result)

        return result

    def visualize(self):
        return None

    def forward(self, X):
        del X
        return None
