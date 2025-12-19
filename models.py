from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Input Model (What Flutter sends for Diagnosis) ---
class DiagnosisRequest(BaseModel):
    # Required for initial filtering
    species: str = Field(..., description="Pet species: 'Dog' or 'Cat'")
    breed: str = Field(..., description="Pet breed, e.g., 'German Shepherd'")
    age: Optional[float] = Field(None, description="Pet age in years (used for context)")
    weight: Optional[float] = Field(None, description="Pet weight in kg/lbs (used for context)")

    # Core data from the user interaction
    reported_symptoms: List[str] = Field(..., description="List of raw symptoms reported or selected.")
    # This stores the answers to follow-up questions. Key=QuestionID/SymptomKey, Value=UserAnswer
    follow_up_answers: Dict[str, str] = Field(default_factory=dict, description="Detailed answers to severity questions.")


# --- Output Model (What FastAPI sends back to Flutter) ---
class DiagnosisResponse(BaseModel):
    diagnosis_found: bool = Field(..., description="True if a probable disease was found.")
    probable_condition: Optional[str] = Field(None, description="The most likely disease.")
    severity_level: str = Field(..., description="MILD, MODERATE, or CRITICAL.")
    recommendation: str = Field(..., description="Action required (e.g., 'Bland diet' or 'IMMEDIATE VET').")
    llm_rationale: str = Field(..., description="The full LLM explanation of why the diagnosis was reached.")

# --- Chat Flow Helper Model (For managing the multi-turn chat) ---
class ChatFlowState(BaseModel):
    next_step: str = Field(..., description="The next required input: 'breed', 'symptoms', or 'follow_up_X'.")
    next_question: Optional[str] = Field(None, description="The text of the next question to display to the user.")