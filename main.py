from fastapi import FastAPI, HTTPException
from models import DiagnosisRequest, DiagnosisResponse, ChatFlowState
from diagnosis_logic import filter_data
from llm_service import diagnose
from typing import List, Dict

app = FastAPI(title="Pet Symptom Checker API")

# --- Initial Flow Endpoint (Gets pet details and returns symptoms list) ---
# NOTE: This is slightly simplified from a chat flow, focusing on data transfer.
# The Flutter app should manage the step-by-step chat state and send ALL
# gathered data in the final POST request below.

@app.get("/data/symptoms/{species}")
def get_symptom_list(species: str) -> List[str]:
    """Provides the full list of available symptom keys for the selected species."""
    # Logic to extract all unique symptom_keys from the JSON data for the species
    symptoms = ["lethargy", "vomiting", "diarrhea", "limping", "thirst_urination", "skin_lesions"] 
    return symptoms 

# --- Final Diagnosis Endpoint (The core of the RAG system) ---
@app.post("/diagnosis", response_model=DiagnosisResponse)
def get_diagnosis(request: DiagnosisRequest):
    """
    Receives all user data (Breed, Symptoms, Follow-up Answers) and returns
    a diagnosis using the RAG model.
    """
    if not request.reported_symptoms:
        raise HTTPException(status_code=400, detail="Must provide at least one symptom.")

    # 1. Filter the database based on initial input
    candidate_data = filter_data(
        request.species, 
        request.breed, 
        request.reported_symptoms
    )

    if not candidate_data:
        return DiagnosisResponse(
            diagnosis_found=False,
            severity_level="LOW",
            recommendation="Could not find a match for the combination of breed and symptoms provided in the system database. Please contact your vet for a professional opinion.",
            llm_rationale="No candidate disease profiles matched the initial filtering criteria."
        )

    # 2. Pass filtered data and user input to the LLM service
    final_diagnosis = diagnose(request, candidate_data)

    # 3. Return the structured result
    return final_diagnosis