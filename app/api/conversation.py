from typing import List, Dict, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversation", tags=["Voice & Conversational AI"])


class MessageHistoryItem(BaseModel):
    role: str
    content: str


class ConversationQueryRequest(BaseModel):
    query: str = Field(..., description="The user's voice or text question")
    language: Optional[str] = Field("en", description="ISO language code: en, hi, te")
    experiment_code: Optional[str] = Field("EXP-004", description="Active experiment identifier")
    history: Optional[List[MessageHistoryItem]] = Field(default=[], description="Recent multi-turn conversation context")


class ActionSuggestion(BaseModel):
    screen: str
    label: str


class ConversationQueryResponse(BaseModel):
    query: str
    response_text: str
    spoken_text: str
    language: str
    experiment_code: str
    category: Optional[str] = "general"
    tool_calls: Optional[List[str]] = []
    visual_state: Optional[str] = "speaking"
    suggested_action: Optional[ActionSuggestion] = None
    follow_up_prompt: Optional[str] = None
    context_snapshot: Dict[str, Any]


@router.post("/query", response_model=ConversationQueryResponse)
def handle_conversation_query(
    request: ConversationQueryRequest = Body(...),
    db: Session = Depends(get_db),
):
    """
    Process conversational and voice queries grounded in actual experiment records and dataset metrics.
    Includes clinical safety disclaimers, multilingual answers, and suggested research actions.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    history_dicts = [
        {"role": h.role, "content": h.content} for h in (request.history or [])
    ]

    result = ConversationService.process_query(
        db=db,
        query=request.query,
        language=request.language or "en",
        experiment_code=request.experiment_code or "EXP-004",
        history=history_dicts,
    )

    return result
