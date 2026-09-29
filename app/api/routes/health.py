import sys
import platform
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.core.database import get_db
from app.schemas.health import HealthCheckResponse
from app.services.quantum_service import quantum_service

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse, tags=["System Health"])
def health_check(db: Session = Depends(get_db)):
    # Verify DB connectivity
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"disconnected: {str(e)}"

    # Check Quantum framework
    qiskit_info = quantum_service.check_qiskit_availability()

    return HealthCheckResponse(
        status="healthy" if db_status == "connected" else "degraded",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        database_status=db_status,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        qiskit_available=qiskit_info["available"],
        qiskit_version=qiskit_info["version"],
        system_info={
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "default_shots": settings.QISKIT_DEFAULT_SHOTS,
            "qiskit_backend": settings.QISKIT_BACKEND,
        },
        timestamp=datetime.utcnow()
    )
