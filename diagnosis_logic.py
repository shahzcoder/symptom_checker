import json
from typing import List, Dict, Optional

# Load the entire JSON dataset once on startup
try:
    with open("data/vet_data.json", "r") as f:
        VET_DATA = json.load(f)
except FileNotFoundError:
    print("Error: data/vet_data.json not found.")
    VET_DATA = []

def filter_data(species: str, breed: Optional[str], symptoms: List[str]) -> List[Dict]:
    """
    Filters the master dataset to find candidate diseases using the 
    simplified Mild/Severe symptom lists.
    """
    candidates = []
    
    # Normalize input symptoms to lowercase for better matching
    user_symptoms = [s.lower().strip() for s in symptoms]
    
    for disease in VET_DATA:
        # Step 1: Filter by Species (Case-insensitive)
        if disease['species'].lower() != species.lower():
            continue
        
        # Step 2: Filter by Breed
        # If breed is provided, check for specific match OR "All"
        if breed and breed.strip():
            breed_list = [b.lower() for b in disease['breeds_affected']]
            # Match if specific breed found or if the disease affects 'All' breeds
            if breed.lower().strip() not in breed_list and "all" not in breed_list:
                continue
        else:
            # If NO breed is provided, prioritize common/general diseases 
            # to give the LLM context for follow-up questions
            # Only includes diseases that affect 'All' breeds
            if "All" not in disease['breeds_affected']:
                continue
            
        # Step 3: Filter by reported symptoms
        # Check if user input exists in either the mild or severe lists
        all_disease_symptoms = [s.lower() for s in (disease['mild_symptoms'] + disease['severe_symptoms'])]
        
        if any(symp in all_disease_symptoms for symp in user_symptoms):
            candidates.append(disease)
            
    return candidates