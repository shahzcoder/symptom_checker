import json
from typing import List, Dict

# Load the entire JSON dataset once on startup
with open("data/vet_data.json", "r") as f:
    VET_DATA = json.load(f)

def filter_data(species: str, breed: str, symptoms: List[str]) -> List[Dict]:
    """Filters the master dataset to find candidate diseases."""
    candidates = []
    
    for disease in VET_DATA:
        # Step 1: Filter by Species
        if disease['species'] != species:
            continue
        
        # Step 2: Filter by Breed (must match specific breed OR be "All")
        if breed not in disease['breeds_affected'] and "All" not in disease['breeds_affected']:
            continue
            
        # Step 3: Filter by reported symptoms (must share at least one symptom)
        disease_symptom_keys = [s['symptom_key'] for s in disease['symptoms']]
        if any(symp in disease_symptom_keys for symp in symptoms):
            candidates.append(disease)
            
    return candidates

# ... other functions to manage the follow-up flow ...