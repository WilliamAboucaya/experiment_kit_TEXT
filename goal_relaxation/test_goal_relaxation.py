import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "shahin-as/bart-large-sentence-compression"

model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def compress_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs, max_length=50, num_beams=5, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

df = pd.read_csv("list_goals.csv")
df["compressed text"] = df["goal"].apply(compress_sentence)
df.to_csv("compressed_goals.csv", index=False)

# Example usage
#sentence = "Anticipate the impact of floods on people"
#compressed_sentence = compress_sentence(sentence)
#print("Original:", sentence)
#print("Compressed:", compressed_sentence)

