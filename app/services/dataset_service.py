import os
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from app.models.dataset import Dataset

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_ROWS = 10
MIN_COLS = 2


class DatasetService:
    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def validate_csv(content: bytes, filename: str) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
        """Validates file type, size, parse integrity, minimum rows and columns."""
        if not filename.lower().endswith(".csv"):
            return False, "Invalid file format. Only CSV files are supported for biomedical data ingestion.", None

        if len(content) == 0:
            return False, "Uploaded CSV file is empty (0 bytes).", None

        if len(content) > MAX_FILE_SIZE_BYTES:
            return False, f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.", None

        try:
            import io
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            return False, f"Malformed CSV file. Could not parse table: {str(e)}", None

        if df.shape[0] < MIN_ROWS:
            return False, f"Dataset has insufficient rows ({df.shape[0]}). Minimum required is {MIN_ROWS}.", None

        if df.shape[1] < MIN_COLS:
            return False, f"Dataset has insufficient columns ({df.shape[1]}). Minimum required is {MIN_COLS}.", None

        return True, None, df

    @staticmethod
    def inspect_dataframe(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        row_count, col_count = df.shape
        missing_count = int(df.isna().sum().sum())
        total_cells = row_count * col_count if (row_count * col_count) > 0 else 1
        missing_ratio = float(missing_count / total_cells)
        duplicate_count = int(df.duplicated().sum())

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        # Class distribution if target_col specified
        class_dist = {}
        class_imbalance = "Balanced"
        detected_target = target_col

        # Auto-detect target column if not provided
        if not detected_target:
            candidates = ["diagnosis", "target", "label", "class", "outcome", "status", "risk"]
            for cand in candidates:
                for c in df.columns:
                    if c.strip().lower() == cand:
                        detected_target = c
                        break
                if detected_target:
                    break

        if detected_target and detected_target in df.columns:
            counts = df[detected_target].dropna().value_counts().to_dict()
            class_dist = {str(k): int(v) for k, v in counts.items()}
            if len(counts) >= 2:
                vals = list(counts.values())
                ratio = max(vals) / (min(vals) if min(vals) > 0 else 1)
                if ratio > 3.0:
                    class_imbalance = "High"
                elif ratio > 1.5:
                    class_imbalance = "Moderate"

        return {
            "row_count": row_count,
            "col_count": col_count,
            "missing_ratio": round(missing_ratio, 4),
            "missing_count": missing_count,
            "duplicate_count": duplicate_count,
            "numeric_features_count": len(numeric_cols),
            "categorical_features_count": len(categorical_cols),
            "feature_names": list(df.columns),
            "target_column": detected_target,
            "class_distribution": class_dist,
            "class_imbalance": class_imbalance,
        }

    @staticmethod
    def analyze_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Calculates detailed data quality metrics, potential ID columns, zero-variance columns, and warnings."""
        row_count, col_count = df.shape
        missing_count = int(df.isna().sum().sum())
        total_cells = row_count * col_count if (row_count * col_count) > 0 else 1
        missing_pct = round((missing_count / total_cells) * 100, 2)
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_count / (row_count if row_count > 0 else 1)) * 100, 2)

        warnings: List[Dict[str, str]] = []

        # 1. Constant / Zero-variance columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            warnings.append({
                "type": "constant_columns",
                "severity": "warning",
                "message": f"{len(constant_cols)} constant column(s) detected with zero variance: {', '.join(constant_cols[:5])}."
            })

        # 2. Near-zero variance columns (numeric only)
        numeric_df = df.select_dtypes(include=["number"])
        near_zero_cols = []
        for c in numeric_df.columns:
            if c not in constant_cols:
                var = numeric_df[c].var()
                if var is not None and not np.isnan(var) and var < 1e-4:
                    near_zero_cols.append(c)

        if near_zero_cols:
            warnings.append({
                "type": "near_zero_variance",
                "severity": "info",
                "message": f"{len(near_zero_cols)} near-zero variance column(s) detected (var < 0.0001)."
            })

        # 3. Potential ID columns
        id_candidates = []
        for c in df.columns:
            c_lower = c.strip().lower()
            if any(term in c_lower for term in ["id", "index", "sample_id", "case_id", "barcode"]) or (df[c].nunique() == row_count and df[c].dtype == "object"):
                id_candidates.append(c)

        if id_candidates:
            warnings.append({
                "type": "potential_id",
                "severity": "info",
                "message": f"Potential sample identifier column(s) detected: {', '.join(id_candidates[:4])}. Exclude these from modeling to prevent leakage."
            })

        # 4. Missingness warning
        if missing_pct > 0:
            severity = "warning" if missing_pct > 5.0 else "info"
            warnings.append({
                "type": "missing_values",
                "severity": severity,
                "message": f"Dataset contains {missing_pct}% missing values across {missing_count} cells."
            })
        else:
            warnings.append({
                "type": "missing_values_clean",
                "severity": "success",
                "message": "✓ Zero missing values detected across all features."
            })

        # 5. Duplicate rows warning
        if duplicate_count > 0:
            warnings.append({
                "type": "duplicate_rows",
                "severity": "warning",
                "message": f"{duplicate_count} duplicate row(s) ({duplicate_pct}%) detected."
            })
        else:
            warnings.append({
                "type": "duplicate_clean",
                "severity": "success",
                "message": "✓ Zero duplicate rows detected."
            })

        # 6. Target & Class imbalance warning
        class_distribution = {}
        class_imbalance = "Balanced"
        if target_col and target_col in df.columns:
            counts = df[target_col].dropna().value_counts().to_dict()
            class_distribution = {str(k): int(v) for k, v in counts.items()}
            if len(counts) >= 2:
                vals = list(counts.values())
                ratio = max(vals) / (min(vals) if min(vals) > 0 else 1)
                if ratio > 3.0:
                    class_imbalance = "High"
                    warnings.append({
                        "type": "class_imbalance",
                        "severity": "warning",
                        "message": f"High class imbalance detected on '{target_col}' (ratio {ratio:.1f}:1). Balanced accuracy evaluation recommended."
                    })
                elif ratio > 1.5:
                    class_imbalance = "Moderate"
                    warnings.append({
                        "type": "class_imbalance",
                        "severity": "info",
                        "message": f"Moderate class imbalance detected on '{target_col}' (ratio {ratio:.1f}:1)."
                    })
                else:
                    warnings.append({
                        "type": "class_balance_good",
                        "severity": "success",
                        "message": f"✓ Balanced class distribution detected on target '{target_col}'."
                    })
        elif target_col:
            warnings.append({
                "type": "target_missing",
                "severity": "warning",
                "message": f"Specified target column '{target_col}' was not found in dataset."
            })

        # Overall quality score calculation
        quality_score = 100
        quality_score -= min(30, int(missing_pct * 3))
        quality_score -= min(20, int(duplicate_pct * 4))
        if constant_cols:
            quality_score -= min(15, len(constant_cols) * 2)
        if class_imbalance == "High":
            quality_score -= 10
        quality_score = max(20, min(100, quality_score))

        return {
            "row_count": row_count,
            "col_count": col_count,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": duplicate_pct,
            "constant_columns": constant_cols,
            "near_zero_variance_columns": near_zero_cols,
            "potential_id_columns": id_candidates,
            "class_distribution": class_distribution,
            "class_imbalance": class_imbalance,
            "quality_score": quality_score,
            "warnings": warnings,
            "leakage_safe": True
        }

    @staticmethod
    def get_preview(df: pd.DataFrame, n: int = 10) -> Dict[str, Any]:
        """Returns the first n rows as records and data types dictionary."""
        head_df = df.head(n)
        # Convert NaN to None for JSON serializability
        records = head_df.replace({np.nan: None}).to_dict(orient="records")
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return {
            "columns": list(df.columns),
            "dtypes": dtypes,
            "rows": records,
            "total_rows": df.shape[0],
            "total_cols": df.shape[1],
            "preview_count": len(records),
        }

    @staticmethod
    def get_all_datasets(db: Session) -> List[Dataset]:
        return db.query(Dataset).order_by(Dataset.created_at.desc()).all()

    @staticmethod
    def get_dataset_by_id(db: Session, dataset_id: int) -> Optional[Dataset]:
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()

    @staticmethod
    def load_dataframe(dataset: Dataset) -> pd.DataFrame:
        if not os.path.exists(dataset.file_path):
            raise FileNotFoundError(f"Dataset file not found on disk at {dataset.file_path}")
        return pd.read_csv(dataset.file_path)


dataset_service = DatasetService()
