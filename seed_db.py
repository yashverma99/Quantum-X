import os
import pandas as pd
from app.core.database import SessionLocal, init_db
from app.models.dataset import Dataset
from app.services.dataset_service import dataset_service

def seed():
    init_db()
    db = SessionLocal()
    try:
        sample_path = os.path.join(os.path.dirname(__file__), "datasets", "tcga_brca_sample.csv")
        if not os.path.exists(sample_path):
            print("Sample dataset not found, skipping seed.")
            return

        existing = db.query(Dataset).filter(Dataset.name == "TCGA-BRCA (Sample)").first()
        if existing:
            print("Dataset already seeded.")
            return

        df = pd.read_csv(sample_path)
        with open(sample_path, "rb") as f:
            file_bytes = f.read()
        file_hash = dataset_service.compute_file_hash(file_bytes)
        analysis = dataset_service.inspect_dataframe(df, target_col="diagnosis")

        record = Dataset(
            name="TCGA-BRCA (Sample)",
            description="The Cancer Genome Atlas Breast Invasive Carcinoma cohort sample (569 cases, 30 molecular diagnostic features).",
            file_path=sample_path,
            file_hash=file_hash,
            row_count=analysis["row_count"],
            col_count=analysis["col_count"],
            target_column="diagnosis",
            class_distribution=analysis["class_distribution"],
            missing_ratio=analysis["missing_ratio"],
            duplicate_count=analysis["duplicate_count"],
            numeric_features_count=analysis["numeric_features_count"],
            categorical_features_count=analysis["categorical_features_count"],
            feature_names=analysis["feature_names"],
        )
        db.add(record)
        db.commit()
        print(f"Successfully seeded dataset {record.name} (ID: {record.id}).")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
