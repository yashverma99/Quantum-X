import os
import shutil
import io
import pandas as pd
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetResponse
from app.services.dataset_service import dataset_service

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=List[DatasetResponse])
def get_datasets(db: Session = Depends(get_db)):
    return dataset_service.get_all_datasets(db)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = dataset_service.get_dataset_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")
    return dataset


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    target_column: str = Form(None),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV files are supported for biomedical data ingestion."
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_hash = dataset_service.compute_file_hash(content)

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV file contains no data rows.")

    # Save to storage directory
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{file_hash[:12]}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(content)

    # Inspect dataset
    analysis = dataset_service.inspect_dataframe(df, target_column)

    dataset_record = Dataset(
        name=name,
        description=description,
        file_path=file_path,
        file_hash=file_hash,
        row_count=analysis["row_count"],
        col_count=analysis["col_count"],
        target_column=target_column,
        class_distribution=analysis["class_distribution"],
        missing_ratio=analysis["missing_ratio"],
        duplicate_count=analysis["duplicate_count"],
        numeric_features_count=analysis["numeric_features_count"],
        categorical_features_count=analysis["categorical_features_count"],
        feature_names=analysis["feature_names"],
    )
    db.add(dataset_record)
    db.commit()
    db.refresh(dataset_record)

    return dataset_record
