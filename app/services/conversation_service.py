"""
MediQAI Conversation & Voice AI Service
Orchestrates AI Reasoning Model, Project Tools, and Conversational Grounding.
Supports General AI, Project Data Queries, and Hybrid Reasoning.
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.services.ai_reasoning_engine import AIReasoningEngine
from app.services.project_tools import ProjectTools

logger = logging.getLogger("mediqai.conversation")


class ConversationService:
    @staticmethod
    def process_query(
        db: Session,
        query: str,
        language: str = "en",
        experiment_code: Optional[str] = "EXP-004",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Process user voice/text query through the AI Reasoning Engine,
        dispatching to ProjectTools when project data is required,
        or applying general-purpose AI reasoning for concepts, education,
        and open-ended dialogue.
        """
        # 1. Resolve experiment context
        exp = None
        if experiment_code:
            exp = db.query(Experiment).filter(Experiment.experiment_code == experiment_code).first()
        if not exp:
            exp = db.query(Experiment).order_by(Experiment.id.desc()).first()

        exp_id = exp.id if exp else 4
        exp_code = exp.experiment_code if exp else "EXP-004"

        # 2. Process query through the AI Reasoning Engine
        reasoning_result = AIReasoningEngine.process_query(
            db=db,
            query=query,
            language=language or "en",
            experiment_id=exp_id,
            history=history or [],
        )

        # 3. Build snapshot for UI context drawer
        dataset = db.query(Dataset).first()
        ds_name = dataset.name if dataset else "TCGA-BRCA"
        comp_summary = ProjectTools.compare_models(db, exp_id)

        context_snapshot = {
            "dataset": ds_name,
            "experiment": exp_code,
            "classical_balanced_accuracy": comp_summary["classical_baseline"]["balanced_accuracy"],
            "vqc_balanced_accuracy": comp_summary["quantum_vqc"]["balanced_accuracy"],
            "qsvm_balanced_accuracy": comp_summary["quantum_qsvm"]["balanced_accuracy"],
            "category": reasoning_result.get("category", "general"),
            "tool_calls": reasoning_result.get("tool_calls", []),
        }

        return {
            "query": query,
            "response_text": reasoning_result["response_text"],
            "spoken_text": reasoning_result["spoken_text"],
            "language": language or "en",
            "experiment_code": exp_code,
            "category": reasoning_result.get("category", "general"),
            "tool_calls": reasoning_result.get("tool_calls", []),
            "visual_state": reasoning_result.get("visual_state", "speaking"),
            "suggested_action": reasoning_result.get("suggested_action"),
            "follow_up_prompt": reasoning_result.get("follow_up_prompt"),
            "context_snapshot": context_snapshot,
        }
