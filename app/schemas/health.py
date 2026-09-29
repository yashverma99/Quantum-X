from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str
    project_name: str
    version: str
    environment: str
    database_status: str
    python_version: str
    qiskit_available: bool
    qiskit_version: Optional[str] = None
    system_info: Dict[str, Any]
    timestamp: datetime
