import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _to_numpy(x):
    if hasattr(x, "detach"): return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize(importance):
    importance = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
    total = importance.sum()
    return importance / total if total > 1e-12 else np.zeros_like(importance)


def _build_preprocessor(feature_type, columns=None):
    feature_type = np.asarray(feature_type)
    columns = np.arange(len(feature_type)) if columns is None else np.asarray(columns)
    continuous = columns[feature_type[columns] == 0]
    categorical = columns[feature_type[columns] != 0]
    transformers = []

    if len(continuous):
        transformers.append(("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), continuous))

    if len(categorical):
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))

    return ColumnTransformer(transformers, remainder="drop")


def _fit_score(X_train, y_train, X_test, y_test, feature_type, num_classes, columns=None):
    preprocessor = _build_preprocessor(feature_type, columns)

    if num_classes is None:
        model = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
        model.fit(X_train, y_train)
        return r2_score(y_test, model.predict(X_test))

    model = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train.astype(np.int64))
    return balanced_accuracy_score(y_test.astype(np.int64), model.predict(X_test))


def mutual_information_importance(X_train, y_train, feature_type, num_classes, seed=0):
    X = _to_numpy(X_train).astype(np.float64).copy()
    y = _to_numpy(y_train).reshape(-1)
    feature_type = _to_numpy(feature_type).astype(np.int64)
    discrete = feature_type != 0

    for j in range(X.shape[1]):
        missing = np.isnan(X[:, j])
        observed = X[~missing, j]
        if not missing.any(): continue
        if len(observed) == 0: X[missing, j] = 0.0
        elif discrete[j]:
            values, counts = np.unique(observed, return_counts=True)
            X[missing, j] = values[np.argmax(counts)]
        else: X[missing, j] = np.median(observed)

    if num_classes is None:
        importance = mutual_info_regression(X, y, discrete_features=discrete, random_state=seed)
    else:
        importance = mutual_info_classif(X, y.astype(np.int64), discrete_features=discrete, random_state=seed)

    return _normalize(importance)


def marginal_importance(
    X_train, y_train, X_test, y_test, feature_type, num_classes
):
    X_train = _to_numpy(X_train).astype(np.float64)
    X_test = _to_numpy(X_test).astype(np.float64)
    y_train = _to_numpy(y_train).reshape(-1)
    y_test = _to_numpy(y_test).reshape(-1)
    feature_type = _to_numpy(feature_type).astype(np.int64)

    d = X_train.shape[1]
    importance = np.zeros(d, dtype=np.float64)

    for j in range(d):
        xtr = X_train[:, j].copy()
        xte = X_test[:, j].copy()

        if feature_type[j] == 0:
            # continuous: median imputation + standardization
            observed = xtr[~np.isnan(xtr)]
            fill = np.median(observed) if len(observed) else 0.0
            xtr[np.isnan(xtr)] = fill
            xte[np.isnan(xte)] = fill
            mean = xtr.mean()
            std = xtr.std()
            if std < 1e-12:
                std = 1.0
            xtr = ((xtr - mean) / std).reshape(-1, 1)
            xte = ((xte - mean) / std).reshape(-1, 1)

        else:
            # categorical: most-frequent imputation + one-hot
            observed = xtr[~np.isnan(xtr)]
            if len(observed):
                values, counts = np.unique(observed, return_counts=True)
                fill = values[np.argmax(counts)]
            else:
                fill = 0.0
            xtr[np.isnan(xtr)] = fill
            xte[np.isnan(xte)] = fill
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            xtr = encoder.fit_transform(xtr.reshape(-1, 1))
            xte = encoder.transform(xte.reshape(-1, 1))

        if num_classes is None:
            model = LinearRegression()
            model.fit(xtr, y_train)
            score = r2_score(y_test, model.predict(xte))

        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(xtr, y_train.astype(np.int64))
            score = balanced_accuracy_score(y_test.astype(np.int64), model.predict(xte))

        importance[j] = score

    if num_classes is None:
        importance = np.maximum(importance, 0.0)
    else:
        chance = 1.0 / num_classes
        importance = np.maximum(importance - chance, 0.0)

    return _normalize(importance)




def loco_importance(X_train, y_train, X_test, y_test, feature_type, num_classes):
    X_train = _to_numpy(X_train).astype(np.float64)
    X_test = _to_numpy(X_test).astype(np.float64)
    y_train = _to_numpy(y_train).reshape(-1)
    y_test = _to_numpy(y_test).reshape(-1)
    feature_type = _to_numpy(feature_type).astype(np.int64)

    d = X_train.shape[1]

    preprocessor = _build_preprocessor(feature_type)
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)
    feature_columns = {}
    continuous = np.where(feature_type == 0)[0]
    categorical = np.where(feature_type != 0)[0]
    offset = 0

    for j in continuous:
        feature_columns[j] = np.array([offset], dtype=np.int64)
        offset += 1

    if len(categorical):
        cat_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = cat_pipeline.named_steps["onehot"]
        for j, categories in zip(categorical, encoder.categories_):
            width = len(categories)
            feature_columns[j] = np.arange(offset, offset + width, dtype=np.int64)
            offset += width

    def fit_score(Xtr_i, Xte_i):
        if num_classes is None:
            model = LinearRegression()
            model.fit(Xtr_i, y_train)
            prediction = model.predict(Xte_i)
            return r2_score(y_test, prediction)

        model = LogisticRegression(max_iter=1000)
        model.fit(Xtr_i, y_train.astype(np.int64))
        prediction = model.predict(Xte_i)

        return balanced_accuracy_score(y_test.astype(np.int64), prediction)

    baseline = fit_score(Xtr, Xte)
    importance = np.zeros(d, dtype=np.float64)
    all_columns = np.arange(Xtr.shape[1])

    for j in range(d):
        removed = feature_columns[j]
        keep = np.setdiff1d(all_columns, removed, assume_unique=True)
        reduced_score = fit_score(Xtr[:, keep], Xte[:, keep])
        importance[j] = baseline - reduced_score

    return _normalize(importance)


def permutation_importance(X_train, y_train, X_test, y_test, feature_type, num_classes, seed=0, n_repeats=3):
    X_train = _to_numpy(X_train).astype(np.float64)
    X_test = _to_numpy(X_test).astype(np.float64)
    y_train = _to_numpy(y_train).reshape(-1)
    y_test = _to_numpy(y_test).reshape(-1)
    feature_type = _to_numpy(feature_type).astype(np.int64)

    preprocessor = _build_preprocessor(feature_type)
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)

    if num_classes is None:
        model = LinearRegression()
        model.fit(Xtr, y_train)
        baseline = r2_score(y_test, model.predict(Xte))
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(Xtr, y_train.astype(np.int64))
        baseline = balanced_accuracy_score(y_test.astype(np.int64), model.predict(Xte))

    feature_columns = {}
    continuous = np.where(feature_type == 0)[0]
    categorical = np.where(feature_type != 0)[0]
    offset = 0

    for j in continuous:
        feature_columns[j] = np.array([offset], dtype=np.int64)
        offset += 1

    if len(categorical):
        encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        for j, categories in zip(categorical, encoder.categories_):
            width = len(categories)
            feature_columns[j] = np.arange(offset, offset + width, dtype=np.int64)
            offset += width

    rng = np.random.default_rng(seed)
    importance = np.zeros(X_train.shape[1], dtype=np.float64)

    for j in range(X_train.shape[1]):
        scores = []
        columns = feature_columns[j]

        for _ in range(n_repeats):
            permutation = rng.permutation(Xte.shape[0])
            Xte_permuted = Xte.copy()
            Xte_permuted[:, columns] = Xte[permutation][:, columns]
            prediction = model.predict(Xte_permuted)

            if num_classes is None:
                scores.append(r2_score(y_test, prediction))
            else:
                scores.append(balanced_accuracy_score(y_test.astype(np.int64), prediction))

        importance[j] = baseline - np.mean(scores)

    return _normalize(importance)