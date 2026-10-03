import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "MediQAI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    API_V1_STR: str = "/api"

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:////tmp/mediqai.db" if os.getenv("VERCEL") == "1" else "sqlite:///./mediqai.db",
    )

    # Security & CORS
    SECRET_KEY: str = "mediqai_research_secret_key_change_in_production"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Quantum Simulator Configuration
    QISKIT_BACKEND: str = "aer_simulator"
    QISKIT_DEFAULT_SHOTS: int = 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
