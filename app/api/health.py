import sys
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.core.database import get_db
from app.services.quantum_service import quantum_service

router = APIRouter()


@router.get("/health", tags=["System Health"])
def get_health(db: Session = Depends(get_db)):
    # 1. Database connection check
    database_status = "not_connected"
    try:
        db.execute(text("SELECT 1"))
        database_status = "configured"
    except Exception:
        database_status = "not_connected"

    # 2. Quantum simulator real execution check
    q_check = quantum_service.check_qiskit_availability()
    quantum_simulator = "available" if q_check.get("available") else "unavailable"

    # 3. Overall status
    status = "ok" if database_status == "configured" else "degraded"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return {
        "status": status,
        "service": "MediQAI Backend",
        "version": settings.VERSION,
        "python": python_version,
        "quantum_simulator": quantum_simulator,
        "database": database_status
    }
