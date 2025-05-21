import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import RobertaModel, RobertaTokenizer
import torch
from adjustText import adjust_text
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset_balanced_90.csv')

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Function to get RoBERTa embeddings
def get_roberta_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use mean of last hidden states as the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)

    return np.array(embeddings)


# Get embeddings for each verb with its context (for better contextual understanding)
texts = [f"{row['verb']}: {row['context']}" for _, row in df.iterrows()]
embeddings = get_roberta_embeddings(texts)

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

# Create DataFrame for plotting
plot_df = pd.DataFrame({
    'verb': df['verb'],
    'goal_type': df['goal_type'],
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1]
})

# Color mapping
color_map = {
    'ACHIEVE': '#4daf4a',  # green
    'AVOID': '#e41a1c',  # red
    'MAINTAIN': '#377eb8',  # blue
    'CEASE': '#984ea3'  # purple
}

# Create plot
plt.figure(figsize=(14, 12))
scatter = sns.scatterplot(
    data=plot_df,
    x='x', y='y',
    hue='goal_type',
    palette=color_map,
    s=100,
    alpha=0.8
)

# Add labels for some points
texts = []
for i, row in plot_df.iterrows():
    #if i % 2 == 0:  # Label every 5th verb
    texts.append(plt.text(row['x'], row['y'], row['verb'], fontsize=9))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

plt.title('RoBERTa embeddings visualization (t-SNE)', fontsize=14)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(title='Goal Types', title_fontsize=12, fontsize=11)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()