import json
import os
from typing import List, Dict
from models import DiagnosisRequest, DiagnosisResponse
from fastapi import HTTPException
from groq import Groq, APIError

# --- 1. Groq Client Setup ---
try:
    # Groq client will automatically look for the GROQ_API_KEY environment variable
    client = Groq()
except Exception as e:
    print(f"Error initializing Groq client. Ensure GROQ_API_KEY is set. Error: {e}")
    client = None 

# --- 2. Master Prompt Generation (RAG) ---

def generate_diagnosis_prompt(request: DiagnosisRequest, candidate_data: List[Dict]) -> str:
    """Creates the RAG prompt for the LLM based on candidate data."""
    
    json_subset_str = json.dumps(candidate_data, indent=2)
    
    # Format user symptoms and any existing follow-up answers for context
    symptom_context = ", ".join(request.reported_symptoms)
    follow_up_context = "\n".join([f"- {k}: {v}" for k, v in request.follow_up_answers.items()])

    # master prompt updated to include the conversational flow and mild/severe logic
    MASTER_PROMPT = f"""
    **INPUT DATA:**
    A. Pet Profile:
    Species: {request.species}
    Breed: {request.breed if request.breed else 'NOT PROVIDED'}
    Age: {request.age if request.age else 'NOT PROVIDED'}
    
    B. User Reported Symptoms:
    {symptom_context}

    C. Existing Follow-up Answers:
    {follow_up_context if follow_up_context else 'None'}
    
    D. Structured Disease Database (JSON Subset):
    {json_subset_str}
    ---
    
    **TASK:**
    You are a Veterinary Diagnostic Assistant. Analyze the input above using the following rules:

    1. MISSING INFO: If Breed or Age is 'NOT PROVIDED', your response MUST politely ask for that information before giving a diagnosis.
    2. SYMPTOM MATCHING: 
       - Compare the User's symptoms against 'mild_symptoms' and 'severe_symptoms' in the database.
       - If only mild symptoms are present, ASK 1-2 follow-up questions from the 'severe_symptoms' list of the suspected disease (e.g., "Are you seeing any bloating or bloody diarrhea?").
    3. FINAL DIAGNOSIS: If Breed, Age, and enough symptoms are known, provide:
       - Possible Predicted Disease: [Name from JSON]
       - Preventions: [Specific preventions from JSON]
       - Mandatory Closing: "Please consult a vet for better diagnosis and timely treatment."
    """
    return MASTER_PROMPT

# --- 3. Groq API Call Function ---

def diagnose(request: DiagnosisRequest, candidate_data: List[Dict]) -> DiagnosisResponse:
    """The main function to call the LLM and parse the result."""
    
    if client is None:
        raise APIError("Groq client not initialized. Check GROQ_API_KEY environment variable.")
        
    prompt = generate_diagnosis_prompt(request, candidate_data)

    system_message = (
        "You are a helpful and professional Veterinary Diagnostic Assistant. "
        "Use provided context to guide users. Always be empathetic but maintain clinical accuracy. "
        "Always recommend professional medical consultation."
    )
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1 # Low temperature for factual consistency
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Check if the AI provided a final diagnosis or just follow-up questions
        is_final = "Predicted Disease" in response_text or "Preventions" in response_text
        
        # Determine Severity based on keywords in the LLM's logic
        severity = "UNKNOWN"
        if "SEVERE" in response_text.upper() or "EMERGENCY" in response_text.upper():
            severity = "SEVERE"
        elif "MILD" in response_text.upper() or "MODERATE" in response_text.upper():
            severity = "MILD/MODERATE"

        # Extract probable condition name if present
        probable_condition = "Awaiting Information"
        if is_final:
            try:
                # Simple split to find the condition name
                probable_condition = response_text.split("Predicted Disease:")[-1].split("\n")[0].strip()
            except:
                probable_condition = "Analysis Provided"

        return DiagnosisResponse(
            diagnosis_found=is_final,
            probable_condition=probable_condition,
            severity_level=severity,
            recommendation=response_text,
            llm_rationale=response_text
        )

    except APIError as e:
        print(f"Groq API Error: {e}")
        raise APIError(f"Failed to connect to Groq API: {e}")
    except Exception as e:
        print(f"General Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")