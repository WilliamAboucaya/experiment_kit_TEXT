import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# Load the model
model_path = './roberta-large-goal-type-classification'

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval() # set the model to evaluation mode


# Load triples
file_path = "triples.csv"
df = pd.read_csv(file_path)

df["text"] = df.apply(lambda row: f"{row['SUBJECT']} {row['PREDICATE']} {row['OBJECT']}", axis=1)

# Tokenization
inputs = tokenizer(df["text"].tolist(), padding=True, truncation=True, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).tolist()

# Define label mapping
label_map = {0: "ACHIEVE", 1: "MAINTAIN", 2: "AVOID", 3: "CEASE"}

print("\nPredictions:")
for pred in predictions:
    print(pred)

# Convert predictions to labels
df["predicted_label"] = [label_map[pred] for pred in predictions]

# Save results
output_file = "classified_triples.csv"
df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")