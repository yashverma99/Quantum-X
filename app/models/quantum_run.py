from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class QuantumRun(Base):
    __tablename__ = "quantum_runs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True, index=True)
    model_name = Column(String(100), default="VQC")
    backend = Column(String(100), default="Qiskit Aer Simulator")
    backend_name = Column(String(100), default="aer_simulator")
    qubits = Column(Integer, default=4)
    circuit_depth = Column(Integer, default=2)
    encoding = Column(String(50), default="Angle Encoding")
    encoding_method = Column(String(50), default="angle_encoding")
    ansatz = Column(String(100), default="RealAmplitudes")
    optimizer = Column(String(50), default="COBYLA")
    iterations = Column(Integer, default=50)
    final_loss = Column(Float, nullable=True)
    shots = Column(Integer, default=1024)
    num_parameters = Column(Integer, default=0)
    noise_level = Column(String(50), default="OFF")
    
    # Genuine performance metrics
    accuracy = Column(Float, nullable=True)
    balanced_accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    
    training_time_ms = Column(Float, default=0.0)
    inference_time_ms = Column(Float, default=0.0)
    confusion_matrix = Column(JSON, nullable=True)
    roc_curve_data = Column(JSON, nullable=True)
    circuit_summary = Column(JSON, nullable=True)
    circuit_qasm = Column(Text, nullable=True)
    circuit_text = Column(Text, nullable=True)
    validation_status = Column(String(50), default="VALIDATED")
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", backref="quantum_runs")
    model = relationship("ModelRecord", back_populates="quantum_run")
