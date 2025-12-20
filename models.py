from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Input Model ---
class DiagnosisRequest(BaseModel):
    # species remains required to start the filter
    species: str = Field(..., description="Pet species: 'Dog' or 'Cat'")
    
    # breed and age are now Optional to allow the conversational start
    breed: Optional[str] = Field(None, description="Pet breed, e.g., 'German Shepherd'")
    age: Optional[str] = Field(None, description="Pet age (e.g., '2 years' or '6 months')")
    
    weight: Optional[float] = Field(None, description="Pet weight (optional context)")

    # Core data from the user interaction
    reported_symptoms: List[str] = Field(..., description="List of symptoms reported.")
    
    # Key=SymptomKey, Value=User's specific detail or answer
    follow_up_answers: Dict[str, str] = Field(default_factory=dict, description="Detailed user answers.")


# --- Output Model ---
class DiagnosisResponse(BaseModel):
    # diagnosis_found will be False if the LLM is still asking follow-up questions
    diagnosis_found: bool = Field(..., description="True if final diagnosis, False if still asking questions.")
    probable_condition: Optional[str] = Field(None, description="The likely disease if found.")
    severity_level: str = Field(..., description="MILD, MODERATE, CRITICAL, or UNKNOWN.")
    
    # This recommendation field will now hold the LLM's "Follow-up Question" 
    # OR the final "Diagnosis + Preventions"
    recommendation: str = Field(..., description="The chat response for the user.")
    
    llm_rationale: str = Field(..., description="The full reasoning from Groq.")

# --- Chat Flow Helper Model ---
class ChatFlowState(BaseModel):
    next_step: str = Field(..., description="Current state: 'gathering_info' or 'final_diagnosis'.")
    next_question: Optional[str] = Field(None, description="The specific question for the user.")