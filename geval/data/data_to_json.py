import json
from pathlib import Path
from datasets import load_dataset

# Reference: https://huggingface.co/datasets/vibrantlabsai/amnesty_qa
dataset = load_dataset("explodinggradients/amnesty_qa", "english_v3")
eval_data = dataset["eval"]


output_dir = Path(__file__) 
output_dir.mkdir(exist_ok=True)

# Convert to simple dictionaries with only 4 fields
serializable_data = []
for idx, sample in enumerate(eval_data):
    serializable_data.append({
        "input": sample["user_input"],
        "actual_output": sample["response"],
        "expected_output": sample["reference"],
        "retrieval_context": sample["retrieved_contexts"]
    })
    
output_file = output_dir / "amnesty_qa_test_cases.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(serializable_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(serializable_data)} test cases to: {output_file}")


import json
from pathlib import Path
import pandas as pd

# Load the CSV file
df = pd.read_csv('Rag-HumanWritten-Test-Cases-12122025.csv')

serializable_data = []

for i in range(len(df)):
    serializable_data.append({
        "input": df.iloc[i]["Input"],
        "actual_output": df.iloc[i]["Output"],
        "expected_output": df.iloc[i]["Expected"],
        "retrieval_context": [df.iloc[i]["Context"]],
        "additional_metadata": {
            "comments": df.iloc[i]["Comments"],
            "tags": df.iloc[i]["Tags"],
            "source": df.iloc[i]["From"]
        }
    })

# 保存为 JS

output_file =  "Rag_HumanWritten_test_cases_12122025.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(serializable_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(serializable_data)} test cases to: {output_file}")