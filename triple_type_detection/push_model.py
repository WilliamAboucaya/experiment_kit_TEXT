from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Define the model name and the folder containing the model files
model_name = "roberta-large-goal-type-classification"
model_folder = f"./{model_name}"
repo_id = f"shahin-as/{model_name}"

# Create the repository on Hugging Face Hub (if not already created)
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Load the fine-tuned model and tokenizer
model = RobertaForSequenceClassification.from_pretrained(model_folder)
tokenizer = RobertaTokenizer.from_pretrained(model_folder)

# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

# Upload the entire model folder (including model.safetensors and other files)
upload_folder(
    folder_path=model_folder,
    path_in_repo=".",
    repo_id=repo_id,
    repo_type="model")

print("Model and all files uploaded successfully!")
