from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from models import DiagnosisRequest, DiagnosisResponse
from diagnosis_logic import filter_data
from llm_service import diagnose

app = FastAPI(title="Smart Paws: Conversational Vet AI")

# --- FIX 1: Add CORS Middleware ---
# This prevents the "Connection failed" error in Flutter Web/FlutterFlow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your FlutterFlow app to access the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data/symptoms/{species}")
def get_symptom_list(species: str) -> List[str]:
    """Returns unique symptom keys for the UI to suggest based on species."""
    # Common symptoms across the diseases we defined
    dog_symptoms = ["lethargy", "vomiting", "diarrhea", "limping", "fever", "coughing", "thirst"]
    cat_symptoms = ["lethargy", "vomiting", "diarrhea", "sneezing", "drooling", "eye discharge"]
    
    return dog_symptoms if species.lower() == "dog" else cat_symptoms

@app.post("/diagnosis", response_model=DiagnosisResponse)
async def get_diagnosis(request: DiagnosisRequest):
    """
    Main Endpoint:
    1. Receives current chat state (Species, Breed, Age, Symptoms).
    2. Filters the JSON database for relevant diseases.
    3. Calls Groq to either ask follow-up questions or provide a final diagnosis.
    """
    if not request.reported_symptoms:
        raise HTTPException(status_code=400, detail="Please describe your pet's symptoms.")

    # 1. Retrieve relevant diseases from JSON (even if breed is missing)
    candidate_data = filter_data(
        request.species, 
        request.breed, 
        request.reported_symptoms
    )

    # 2. Use LLM to manage the conversation flow (Asking info vs. Diagnosing)
    try:
        final_result = diagnose(request, candidate_data)
        return final_result
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during diagnosis.")