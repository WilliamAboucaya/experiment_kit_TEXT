import torch
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from evaluate import load
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


print(torch.cuda.is_available())

print(torch.cuda.get_device_name(0))

assert torch.cuda.is_available()

# Load dataset
dataset = load_dataset("sentence-transformers/sentence-compression")

sari_metric = load("sari")
# bleu_metric = load("bleu")
rouge_metric = load("rouge")

# Load model and tokenizer
model_checkpoint = "facebook/bart-large"
model_name = model_checkpoint.split("/")[-1]
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
batch_size = 8

df_dataset = pd.DataFrame(dataset["train"][:])

trainer_name = f"{model_name}-sentence-compression"

# Preprocessing function
def preprocess_data(df_dataset):
    inputs = df_dataset["text"]  # Input sentences
    targets = df_dataset["simplified"]  # Target compressed sentences
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_data, batched=True)

print("OK preprocessing dataset")

# Set format for PyTorch (to train with PyTorch)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print(tokenized_dataset)

# Splitting the dataset
#tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1) # 10% of the data is reserved for evaluation
#train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)  # 10% reserved for evaluation
#train_dataset = train_test_split["train"]
#eval_dataset = train_test_split["test"]

# Splitting the dataset into train, validation, and test
train_val_split = tokenized_dataset["train"].train_test_split(test_size=0.2)  # 20% for validation + test
val_test_split = train_val_split["test"].train_test_split(test_size=0.5)     # Split validation + test evenly

train_dataset = train_val_split["train"]  # 80% for training
val_dataset = val_test_split["train"]     # 10% for validation
test_dataset = val_test_split["test"]     # 10% for testing

print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
print("Test dataset:", len(test_dataset))


# --> Hyperparameters and configurations for training
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./scratch/{trainer_name}",         # Directory to save model checkpoints
    #evaluation_strategy="epoch",
    #save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    #device_map="auto",  # Automatically distribute the workload
    #gradient_accumulation_steps=2,                  # Simulate larger batch size by accumulating gradients
    load_best_model_at_end=True,                    # To ensure that the best-performing model (according to metric_for_best_model) is loaded after training
    learning_rate=2e-5,                             # A conservative learning rate
    per_device_train_batch_size=batch_size,         # Training batch size
    per_device_eval_batch_size=batch_size,          # Evaluation batch size
    num_train_epochs=5,                             # Number of epochs (for training)
    save_safetensors=True,                          # Save model checkpoints using the more efficient safetensors format, improving safety and speed
    weight_decay=0.01,                              # Weight decay for regularization (to prevent overfitting)
    save_total_limit=3,                             # Limit number of checkpoints to avoid excessive disk usage
    logging_dir="./logs",                           # Directory for logs
    logging_steps=500,                              # Log every 500 steps
    fp16=True,                                      # Use mixed precision if a GPU is available (for faster computations and for reducing memory usage)
    report_to="none",                               # Disable third-party integrations
    predict_with_generate=True,                     # Generation of text outputs during evaluation (to compute metrics like BLEU, ROUGE or SARI), not just predicted token probabilities
    include_inputs_for_metrics=True,                # To ensure that the original inputs are available for use in custom metrics like SARI
    metric_for_best_model="sari_penalized",         # Use sari_penalized to track and select the best-performing model
)


# to pad sequences dynamically during training, which helps reduce memory usage
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)


# Evaluates performance using custom metrics
# def compute_metrics(eval_pred_inputs):
#     predictions, labels, inputs = eval_pred_inputs
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     decoded_labels_as_lists = [[decoded_label] for decoded_label in decoded_labels]
#
#     inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
#     decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
#
#     # Compute SARI
#     result = sari_metric.compute(sources=decoded_inputs, predictions=decoded_preds, references=decoded_labels_as_lists)
#
#     # Add mean generated length
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     input_lens = [np.count_nonzero(init != tokenizer.pad_token_id) for init in inputs]
#     result["gen_len"] = np.mean(prediction_lens)
#
#     len_ratios = [i / j for i, j in zip(input_lens, prediction_lens)]
#     lens_reduced_enough = [1 if len_ratio > 4 / 3 else 0 for len_ratio in len_ratios]
#     mean_len_ratio = np.mean(len_ratios)
#     mean_lens_reduced_enough = np.mean(lens_reduced_enough)
#
#     # Add ROUGE scores
#     rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result["rouge1"] = rouge_results["rouge1"].mid.fmeasure * 100
#     result["rouge2"] = rouge_results["rouge2"].mid.fmeasure * 100
#     result["rougeL"] = rouge_results["rougeL"].mid.fmeasure * 100
#
#     # Add modified SARI score
#     result["sari_penalized"] = result["sari"] * mean_lens_reduced_enough
#     result["mean_len_ratio"] = mean_len_ratio
#     result["mean_lens_reduced_enough"] = mean_lens_reduced_enough
#
#     # Round results for readability
#     return {k: round(v, 4) for k, v in result.items()}


# Evaluates performance using SARI and sari_penalized
def compute_metrics(eval_pred_inputs):
    predictions, labels, inputs = eval_pred_inputs
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels_as_lists = [[decoded_label] for decoded_label in decoded_labels]

    inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    # Compute SARI
    result = sari_metric.compute(sources=decoded_inputs, predictions=decoded_preds, references=decoded_labels_as_lists)
    
    # Add mean generated length
    #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    #input_lens = [np.count_nonzero(init != tokenizer.pad_token_id) for init in inputs]
    #len_ratios = [i / j for i, j in zip(input_lens, prediction_lens)]
    #lens_reduced_enough = [1 if len_ratio > 4 / 3 else 0 for len_ratio in len_ratios]
    #mean_len_ratio = np.mean(len_ratios)
    #mean_lens_reduced_enough = np.mean(lens_reduced_enough)
    #result["gen_len"] = np.mean(prediction_lens)

    #len_ratios = [i / j for i, j in zip(input_lens, prediction_lens)]
    #lens_reduced_enough = [1 if len_ratio > 4 / 3 else 0 for len_ratio in len_ratios]
    #mean_len_ratio = np.mean(len_ratios) # Autre possibilité compter le nb de ratios inférieurs à 1.33 et pénaliser le score en fonction
    #mean_lens_reduced_enough = np.mean(lens_reduced_enough)


    # Add ROUGE scores
    #rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    #result["rouge1"] = rouge_results["rouge1"].mid.fmeasure * 100
    #result["rouge2"] = rouge_results["rouge2"].mid.fmeasure * 100
    #result["rougeL"] = rouge_results["rougeL"].mid.fmeasure * 100

    # Add penalized SARI score
    #result["sari_penalized"] = result["sari"] * mean_lens_reduced_enough
    #result["mean_len_ratio"] = mean_len_ratio
    #result["mean_lens_reduced_enough"] = mean_lens_reduced_enough

    # Round results for readability
    #return {k: round(v, 4) for k, v in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    input_lens = [np.count_nonzero(init != tokenizer.pad_token_id) for init in inputs]
    len_ratios = [i / j for i, j in zip(input_lens, prediction_lens)]
    lens_reduced_enough = [1 if len_ratio > 4 / 3 else 0 for len_ratio in len_ratios]
    mean_len_ratio = np.mean(len_ratios)
    mean_lens_reduced_enough = np.mean(lens_reduced_enough)
    
    # Compute ROUGE scores
    rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    if isinstance(rouge_results["rouge1"], float):  # Handle float values directly
        result["rouge1"] = rouge_results["rouge1"] * 100
        result["rouge2"] = rouge_results["rouge2"] * 100
        result["rougeL"] = rouge_results["rougeL"] * 100
    else:  # Handle objects with 'mid.fmeasure' attributes
        result["rouge1"] = rouge_results["rouge1"].mid.fmeasure * 100
        result["rouge2"] = rouge_results["rouge2"].mid.fmeasure * 100
        result["rougeL"] = rouge_results["rougeL"].mid.fmeasure * 100

    # Add penalized SARI score
    result["sari_penalized"] = result["sari"] * mean_lens_reduced_enough
    result["mean_len_ratio"] = mean_len_ratio
    result["mean_lens_reduced_enough"] = mean_lens_reduced_enough

    # Add generated length
    #result["gen_len"] = np.mean(prediction_lens)

    # Round results
    return {k: round(v, 4) for k, v in result.items()}


# --> Define Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train(resume_from_checkpoint=f"./scratch/{trainer_name}/checkpoint-90000")
#trainer.train(resume_from_checkpoint=True)
#trainer.train()

print("With validation set:")
validation_results = trainer.evaluate()
pprint(validation_results)

print("With test set:")
#test_results = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix='test')
test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
pprint(test_results)

# Save the model
trainer.save_model(f"./{trainer_name}")
print("Model training and saving completed.")

print("\nvalidation KEY:")
print(validation_results.keys())


# Load the template
with open('template_model_card.md', 'r', encoding="utf-8") as template_file:
    template = template_file.read()

# Replace placeholders with actual values
model_card = template.format(
    sari_valid=round(validation_results["eval_sari"], 2),
    sari_test=round(test_results["test_sari"], 2),
    sari_penalized_valid=round(validation_results["eval_sari_penalized"], 2),
    sari_penalized_test=round(test_results["test_sari_penalized"], 2),
    
    rouge1_valid=round(validation_results["eval_rouge1"], 2),
    rouge2_valid=round(validation_results["eval_rouge2"], 2),
    rougel_valid=round(validation_results["eval_rougeL"], 2),
    rouge1_test=round(test_results["test_rouge1"], 2),
    rouge2_test=round(test_results["test_rouge2"], 2),
    rougel_test=round(test_results["test_rougeL"], 2),
)

with open(f"./{trainer_name}/README.md", "w", encoding="utf-8") as model_card_file:
    model_card_file.write(model_card)

log_history = trainer.state.log_history
#x = sorted(list({log["step"] for log in log_history}))
x = sorted(list({log["step"] for log in log_history if "step" in log}))

# Extract epochs, ensuring logs with 'epoch' exist
#x = sorted(list({log["epoch"] for log in log_history if "epoch" in log}))

# Extract losses for training and evaluation, aligned with epochs
#y1 = [log["loss"] if "loss" in log else log["train_loss"] for log in list(filter(lambda log: ("loss" in log) or ("train_loss" in log), log_history))]
#y2 = [log["eval_loss"] for log in list(filter(lambda log: "eval_loss" in log, log_history))]
y1 = [log.get("loss", log.get("train_loss")) for log in log_history if "loss" in log or "train_loss" in log]
y2 = [log["eval_loss"] for log in log_history if "eval_loss" in log]


# Check lengths and truncate if mismatched
#if len(x) < len(y1) or len(x) < len(y2):
#    print(f"log_history: {log_history}")
#    y1 = y1[:len(x)]
#    y2 = y2[:len(x)]

# Plot losses
#fig, ax = plt.subplots()
#ax.plot(x, y1, 'r', label="train_loss")
#ax.plot(x, y2, 'g', label="eval_loss")
#ax.set_xlabel("Epoch", fontsize='large')
#ax.set_xlabel("Step", fontsize='large')
#ax.set_ylabel("Loss", fontsize='large')
#ax.legend()
#plt.tight_layout()

# Truncate to match lengths
min_len = min(len(x), len(y1), len(y2))
x, y1, y2 = x[:min_len], y1[:min_len], y2[:min_len]

fig, ax = plt.subplots()
ax.plot(x, y1, 'r', label="train_loss")
ax.plot(x, y2, 'g', label="eval_loss")
ax.set_xlabel("Step", fontsize='large')
ax.set_ylabel("Loss", fontsize='large')
ax.legend()
plt.tight_layout()
fig.savefig(f"{trainer_name}_loss.eps", format="eps")

#plt.show()
