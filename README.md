# Assignment 2 


## Install dependencies

Run the following command to install dependencies
```
pip install -r requirements.txt
```

## Setup .env file
Create a .env file in the root directory of the project.

Add your API keys as follows:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## Execute data generation script
Run the following command from the project root directory:

```
python src/generation.py
```
This will:

Generate personas (Step 1)
Parse personas using Groq (Step 2)
Run phishing vulnerability audit (Step 3)

## Execute the evaluation script
Run the following command from the project root directory:
```
python src/evaluation.py
```

This will:

Convert parsed personas to CSV
Build the final dataset
Perform bias analysis (gender, domain, region, age)
Generate visualizations (heatmaps)

## Output Files

All generated outputs will be saved in the data/ folder:
```
data/
├── raw_outputs.json
├── parsed_personas.json
├── final_phishing_audit_results.json
├── personas_data.csv
├── final_personas_with_prompt2_iteration.csv
└── final_personas_selected_only.csv
```

# Note:
Due to GPU issue the full code was executued in my google colab where I have boought GPU units for this purpose. Hence, clear git long might not be visible here.

# Author : Infant Mysytica Nikita Fernando - a1962080
