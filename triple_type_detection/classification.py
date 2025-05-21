import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

trainer_name = "roberta-large-goal-type-classification"

# Load dataset
file_path = "dataset_balanced_90.csv"
df = pd.read_csv(file_path)

# Encode labels
label_map = {"ACHIEVE": 0, "MAINTAIN": 1, "AVOID": 2, "CEASE": 3}
df["label"] = df["goal_type"].map(label_map)

# Train-validation-test split
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df[["verb", "context"]].values.tolist(), df["label"].tolist(),
    test_size=0.2, random_state=42, stratify=df["label"]
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    test_size=0.5, random_state=42, stratify=temp_labels
)

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")


def tokenize_function(examples):
    texts = [f"Verb: {v}, Context: {c}" for v, c in examples]
    return tokenizer(texts, padding="max_length", truncation=True)


# Convert to Dataset format
datasets = DatasetDict({
    "train": Dataset.from_dict({
        "text": train_texts, "label": train_labels
    }).map(lambda x: tokenize_function(x["text"]), batched=True),

    "validation": Dataset.from_dict({
        "text": val_texts, "label": val_labels
    }).map(lambda x: tokenize_function(x["text"]), batched=True),

    "test": Dataset.from_dict({
        "text": test_texts, "label": test_labels
    }).map(lambda x: tokenize_function(x["text"]), batched=True)
})

# Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=4)


# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.savefig("confusion_matrix.eps", format='eps')
    plt.close()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Training arguments
training_args = TrainingArguments(
    output_dir=f"./scratch/{trainer_name}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to=["tensorboard"],
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
print("\nStarting training...")
train_result = trainer.train()
eval_results = trainer.evaluate()

# Save the model
trainer.save_model(f"./{trainer_name}")
print("Model training and saving completed.")


# Evaluate on test set
test_results = trainer.evaluate(datasets["test"])


# Plot training loss
def plot_loss():
    logs = trainer.state.log_history
    train_loss = [entry["loss"] for entry in logs if "loss" in entry]
    eval_loss = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(range(1, len(eval_loss) + 1), eval_loss, label='Evaluation Loss', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.savefig("loss_plot.eps", format='eps')
    plt.close()


plot_loss()
print("\nPlots generated.")

# Load and update model card template
with open("template_model_card.md", "r") as f:
    template_model_card = f.read()

# Replace placeholders with actual values
updated_model_card = template_model_card.format(
    accuracy=eval_results["eval_accuracy"],
    precision=eval_results["eval_precision"],
    recall=eval_results["eval_recall"],
    f1=eval_results["eval_f1"],
    test_accuracy=test_results["eval_accuracy"],
    test_precision=test_results["eval_precision"],
    test_recall=test_results["eval_recall"],
    test_f1=test_results["eval_f1"]
)

# Save the updated model card
with open(f"./{trainer_name}/README.md", "w", encoding="utf-8") as model_card_file:
    model_card_file.write(updated_model_card)

print("\nModel card generated.")
