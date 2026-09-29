import os
import io
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset
from app.services.dataset_service import dataset_service

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("")
def get_datasets(db: Session = Depends(get_db)):
    datasets = dataset_service.get_all_datasets(db)
    results = []
    for d in datasets:
        results.append({
            "id": d.id,
            "name": d.name,
            "samples": d.row_count,
            "features": d.col_count,
            "target_column": d.target_column,
            "status": "available",
            "description": d.description,
            "class_distribution": d.class_distribution,
            "missing_ratio": d.missing_ratio,
            "duplicate_count": d.duplicate_count,
            "numeric_features_count": d.numeric_features_count,
            "categorical_features_count": d.categorical_features_count,
            "created_at": d.created_at.isoformat() if d.created_at else None
        })
    return {"datasets": results}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    d = dataset_service.get_dataset_by_id(db, dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")
    return {
        "id": d.id,
        "name": d.name,
        "samples": d.row_count,
        "features": d.col_count,
        "target_column": d.target_column,
        "status": "available",
        "description": d.description,
        "class_distribution": d.class_distribution,
        "missing_ratio": d.missing_ratio,
        "duplicate_count": d.duplicate_count,
        "numeric_features_count": d.numeric_features_count,
        "categorical_features_count": d.categorical_features_count,
        "created_at": d.created_at.isoformat() if d.created_at else None
    }


@router.get("/{dataset_id}/preview")
def get_dataset_preview(dataset_id: int, db: Session = Depends(get_db)):
    d = dataset_service.get_dataset_by_id(db, dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")
    try:
        df = dataset_service.load_dataframe(d)
        preview_data = dataset_service.get_preview(df, n=10)
        return preview_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dataset preview: {str(e)}")


@router.get("/{dataset_id}/quality")
def get_dataset_quality(dataset_id: int, target_column: Optional[str] = None, db: Session = Depends(get_db)):
    d = dataset_service.get_dataset_by_id(db, dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")
    try:
        df = dataset_service.load_dataframe(d)
        target = target_column or d.target_column
        quality_report = dataset_service.analyze_quality(df, target_col=target)
        return quality_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze dataset quality: {str(e)}")


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    target_column: str = Form(None),
    db: Session = Depends(get_db)
):
    content = await file.read()
    valid, err_msg, df = dataset_service.validate_csv(content, file.filename)
    if not valid or df is None:
        raise HTTPException(status_code=400, detail=err_msg or "Invalid CSV file.")

    file_hash = dataset_service.compute_file_hash(content)

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets")
    os.makedirs(save_dir, exist_ok=True)
    # Safe sanitized filename
    safe_filename = f"{file_hash[:12]}_{''.join(c for c in file.filename if c.isalnum() or c in '._-')}"
    file_path = os.path.join(save_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(content)

    analysis = dataset_service.inspect_dataframe(df, target_column)

    dataset_record = Dataset(
        name=name.strip(),
        description=description.strip() if description else f"Uploaded biomedical cohort ({df.shape[0]} rows, {df.shape[1]} features)",
        file_path=file_path,
        file_hash=file_hash,
        row_count=analysis["row_count"],
        col_count=analysis["col_count"],
        target_column=analysis["target_column"],
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

    return {
        "id": dataset_record.id,
        "name": dataset_record.name,
        "samples": dataset_record.row_count,
        "features": dataset_record.col_count,
        "target_column": dataset_record.target_column,
        "status": "Dataset Ready",
        "created_at": dataset_record.created_at.isoformat() if dataset_record.created_at else None
    }
