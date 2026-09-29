import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


class FeatureEngineeringService:
    """Service implementing feature filtering, PCA dimensionality reduction, and quantum encoding mapping."""

    @staticmethod
    def apply_variance_threshold(
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        """Fits VarianceThreshold on Train only and transforms Train and Test."""
        selector = VarianceThreshold(threshold=threshold)
        X_train_filtered = selector.fit_transform(X_train)
        X_test_filtered = selector.transform(X_test)

        support_mask = selector.get_support()
        retained_feature_names = [name for name, retained in zip(feature_names, support_mask) if retained]

        return X_train_filtered, X_test_filtered, retained_feature_names, int(X_train_filtered.shape[1])

    @staticmethod
    def apply_correlation_filter(
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        """
        Removes collinear features where Pearson correlation |r| > threshold.
        Correlation matrix is computed strictly on X_train to prevent leakage.
        """
        if X_train.shape[1] <= 1:
            return X_train, X_test, feature_names, X_train.shape[1]

        corr_matrix = pd.DataFrame(X_train).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        to_keep = [i for i in range(X_train.shape[1]) if i not in to_drop]

        if len(to_keep) == 0:
            to_keep = [0]  # retain at least 1 feature

        X_train_clean = X_train[:, to_keep]
        X_test_clean = X_test[:, to_keep]
        retained_names = [feature_names[i] for i in to_keep if i < len(feature_names)]

        return X_train_clean, X_test_clean, retained_names, int(X_train_clean.shape[1])

    @staticmethod
    def apply_pca(
        X_train: np.ndarray,
        X_test: np.ndarray,
        n_components: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, List[float], float]:
        """
        Fits PCA on X_train only and transforms Train and Test.
        Returns transformed arrays, explained variance ratio per component, and cumulative variance.
        """
        # Ensure n_components does not exceed feature or sample count
        max_possible = min(X_train.shape[0], X_train.shape[1])
        k = max(1, min(n_components, max_possible))

        pca = PCA(n_components=k, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        explained_ratios = [round(float(r), 4) for r in pca.explained_variance_ratio_]
        cumulative_variance = round(float(np.sum(pca.explained_variance_ratio_)), 4)

        return X_train_pca, X_test_pca, explained_ratios, cumulative_variance

    @staticmethod
    def map_to_quantum_state(
        X_train_pca: np.ndarray,
        X_test_pca: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float], str]:
        """
        Maps reduced PCA continuous latent features into bounded quantum angle-ready vector space [0, pi].
        Angle encoding mapping: theta = pi * (x - min) / (max - min) fitted on Train only.
        """
        min_val = np.min(X_train_pca, axis=0)
        max_val = np.max(X_train_pca, axis=0)
        diff = np.where((max_val - min_val) == 0, 1.0, max_val - min_val)

        # Scale to [0, pi] for angle encoding
        X_train_q = np.pi * (X_train_pca - min_val) / diff
        X_test_q = np.pi * (X_test_pca - min_val) / diff
        # Clip to ensure bounds [0, pi]
        X_train_q = np.clip(X_train_q, 0.0, np.pi)
        X_test_q = np.clip(X_test_q, 0.0, np.pi)

        sample_vector = [round(float(val), 4) for val in X_train_q[0]]

        return X_train_q, X_test_q, sample_vector, "Angle Encoding [0, π]"


feature_service = FeatureEngineeringService()
