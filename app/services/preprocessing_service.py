import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class PreprocessingService:
    """Service implementing strict leakage-safe preprocessing pipelines."""

    @staticmethod
    def run_pipeline(
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        missing_strategy: str = "median",
        scaler_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Executes leakage-safe preprocessing:
        1. Separates features X and target y.
        2. Splits into Train and Test partitions BEFORE any transformation.
        3. Fits imputer on TRAIN ONLY; transforms Train and Test.
        4. Fits scaler on TRAIN ONLY; transforms Train and Test.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

        # Exclude obvious non-feature ID columns if all values unique
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_feature_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_feature_cols) == 0:
            raise ValueError("Dataset contains no numeric features for preprocessing.")

        X = df[numeric_feature_cols].copy()
        y_raw = df[target_col].copy()

        # Encode target if categorical
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        class_mapping = {str(cls): int(idx) for idx, cls in enumerate(label_encoder.classes_)}

        # 1. SPLIT BEFORE TRANSFORMATIONS (Prevent Data Leakage)
        stratify = y if len(np.unique(y)) > 1 and min(pd.Series(y).value_counts()) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        # 2. FIT IMPUTER ON TRAIN ONLY
        valid_strategies = ["mean", "median", "most_frequent"]
        strategy = missing_strategy if missing_strategy in valid_strategies else "median"
        imputer = SimpleImputer(strategy=strategy)
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # 3. FIT SCALER ON TRAIN ONLY
        if scaler_type.lower() == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            scaler_params = {
                "min": [float(m) for m in scaler.data_min_[:5]],
                "max": [float(m) for m in scaler.data_max_[:5]],
                "type": "MinMaxScaler(-1, 1)"
            }
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            scaler_params = {
                "mean": [float(m) for m in scaler.mean_[:5]],
                "scale": [float(s) for s in scaler.scale_[:5]],
                "type": "StandardScaler"
            }

        # Class counts
        train_counts = {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()}
        test_counts = {str(k): int(v) for k, v in pd.Series(y_test).value_counts().to_dict().items()}

        return {
            "status": "PREPROCESSED",
            "leakage_safe": True,
            "target_column": target_col,
            "classes": class_mapping,
            "train_samples": int(X_train_scaled.shape[0]),
            "test_samples": int(X_test_scaled.shape[0]),
            "features_count": int(X_train_scaled.shape[1]),
            "feature_names": numeric_feature_cols,
            "test_split_ratio": test_size,
            "random_seed": random_state,
            "missing_strategy": strategy,
            "scaler_type": scaler_type,
            "scaler_parameters": scaler_params,
            "train_class_distribution": train_counts,
            "test_class_distribution": test_counts,
            # Cache processed arrays for feature engineering step
            "_X_train": X_train_scaled,
            "_X_test": X_test_scaled,
            "_y_train": y_train,
            "_y_test": y_test,
        }


preprocessing_service = PreprocessingService()
