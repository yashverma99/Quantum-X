import sys
import platform
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.core.database import get_db
from app.services.quantum_service import quantum_service

router = APIRouter(prefix="/system", tags=["System Status"])


@router.get("/quantum")
def get_quantum_system_status():
    """
    Executes a real 1-qubit Hadamard circuit on the Aer simulator.
    Returns available=True only if the test passes cleanly.
    """
    return quantum_service.check_qiskit_availability()


@router.get("/status")
def get_system_status(db: Session = Depends(get_db)):
    db_status = "not_connected"
    try:
        db.execute(text("SELECT 1"))
        db_status = "configured"
    except Exception:
        db_status = "not_connected"

    q_check = quantum_service.check_qiskit_availability()
    quantum_status = "available" if q_check.get("available") else "unavailable"

    return {
        "status": "ok" if db_status == "configured" else "degraded",
        "service": "MediQAI Backend",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "quantum_simulator": quantum_status,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "healthcare_safety_policy": "Human-in-the-loop enforced. Non-diagnostic research tool."
    }

