import json
import os
from typing import List, Dict
from models import DiagnosisRequest, DiagnosisResponse
from fastapi import HTTPException

# --- 1. Groq Client Setup ---
# Load the API key from environment variables.
# CRITICAL: Do NOT hardcode your key here. 
# Make sure to set the GROQ_API_KEY environment variable.
from groq import Groq, APIError

try:
    # Groq client will automatically look for the GROQ_API_KEY environment variable
    client = Groq()
except Exception as e:
    print(f"Error initializing Groq client. Ensure GROQ_API_KEY is set. Error: {e}")
    # Initialize without key, but the API call will fail without the key/env var.
    client = None 

# --- 2. Master Prompt Generation (RAG) ---

def generate_diagnosis_prompt(request: DiagnosisRequest, candidate_data: List[Dict]) -> str:
    """Creates the RAG prompt for the LLM based on candidate data."""
    
    # 1. Format the Candidate JSON data beautifully for the prompt
    json_subset_str = json.dumps(candidate_data, indent=2)
    
    # 2. Format the user's symptoms and answers
    symptom_context = "\n".join([
        f"- Symptom: {s} | User's detail: {request.follow_up_answers.get(s, 'No further detail.')}"
        for s in request.reported_symptoms
    ])

    # 3. Master Prompt Template (Refined for Groq's System Message)
    MASTER_PROMPT = f"""
    **INSTRUCTIONS:**
    1. Determine Severity: For each reported symptom, match the severity to 'severity_mild' or 'severity_severe' based on the 'User's detail'.
    2. Final Action: If ANY matched symptom is determined to be 'severity_severe' AND 'immediate_vet_flag_severe' is true, the final recommendation MUST be 'CRITICAL / IMMEDIATE VET'.
    3. Output: Provide the diagnosis in a structured, readable format, including the Probable Condition, Severity Level, and Recommendation.

    ---
    **INPUT DATA:**
    A. Pet Profile:
    Species: {request.species}
    Breed: {request.breed}
    Age: {request.age or 'N/A'}
    
    B. Reported Symptoms and Context:
    {symptom_context}
    
    C. Structured Disease Database (JSON Subset):
    {json_subset_str}
    ---
    
    **ANALYSIS AND DIAGNOSIS:**
    """
    return MASTER_PROMPT

# --- 3. Groq API Call Function ---

def diagnose(request: DiagnosisRequest, candidate_data: List[Dict]) -> DiagnosisResponse:
    """The main function to call the LLM and parse the result."""
    
    if client is None:
        raise APIError("Groq client not initialized. Check GROQ_API_KEY environment variable.")
        
    prompt = generate_diagnosis_prompt(request, candidate_data)

    # Define a System Role to enforce structured, clinical output.
    system_message = (
        "You are a highly specialized Veterinary Symptom Analyzer AI. "
        "Your function is to process the structured JSON data provided and generate a diagnosis summary. "
        "STRICTLY ADHERE to the clinical facts, 'severity_mild'/'severity_severe' descriptions, and 'immediate_vet_flag_severe' within the provided JSON data. "
        "Do not guess or introduce diseases not listed in the database. Output a clear, direct analysis."
    )
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            # You can choose a different model if desired, but Llama-3.3-70b is powerful for reasoning.
            model="llama-3.3-70b-versatile",
            temperature=0.1 # Lower temperature for less creativity, more deterministic reasoning
        )
        
        # The LLM's response contains the full diagnosis summary and rationale
        response_text = chat_completion.choices[0].message.content
        
        # --- Post-Processing: Extract Key Fields from the LLM's Text Response ---
        # NOTE: Groq does not currently support JSON output directly for this model,
        # so you need to rely on the prompt to force a structured text output,
        # then parse it back into a structured Python object.
        
        # For simplicity and robustness, we will extract key phrases from the text response.
        
        # You would implement more robust text parsing (e.g., regex) here.
        # For this example, we'll look for keywords.
        
        probable_condition = "Analysis Required"
        severity = "MODERATE"
        recommendation = "Review LLM output for details."
        
        if "CRITICAL" in response_text.upper() or "EMERGENCY" in response_text.upper():
            severity = "CRITICAL"
            recommendation = "EMERGENCY: Seek immediate veterinary care! " + response_text.split("Action:")[-1].strip()
        elif "MILD" in response_text.upper() or "MODERATE" in response_text.upper():
            severity = "MILD/MODERATE"
            recommendation = response_text.split("Action:")[-1].strip()
        
        # Simple extraction of the first mentioned probable condition (if possible)
        try:
            probable_condition = response_text.split("Probable Condition(s):")[-1].split("\n")[0].strip()
        except IndexError:
            pass # Fallback to default

        return DiagnosisResponse(
            diagnosis_found=True,
            probable_condition=probable_condition,
            severity_level=severity,
            recommendation=recommendation,
            llm_rationale=response_text
        )

    except APIError as e:
        print(f"Groq API Error: {e}")
        raise APIError(f"Failed to connect to Groq API or an API error occurred: {e}")
    except Exception as e:
        print(f"General Error during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during LLM processing: {e}")