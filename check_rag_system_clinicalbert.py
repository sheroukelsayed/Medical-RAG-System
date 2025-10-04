# -*- coding: utf-8 -*-
# Pediatric RAG System Evaluation (ClinicalBERT Phase)
# This notebook evaluates retrieved answers using Cosine Similarity and BERTScore

# Install required libraries
!pip install pandas sentence-transformers bert-score scikit-learn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from bert_score import score

# Load the dataset
file_path = '/content/drive/My Drive/pedratic/Clinicalbert/piadetric_QA_with_pinecone.csv'
df = pd.read_csv(file_path)

# Merge relevant columns into a single text column for evaluation
df['merged_question_answer_explanation'] = (
    "Department: " + df['Depertment '].fillna('') + " " +
    df['Question'].fillna('') + " " +
    df['Answer'].fillna('') + " " +
    df['Explanation'].fillna('')
)

# Remove rows with missing merged data
df = df.dropna(subset=['merged_question_answer_explanation'])

# Initialize embedding model (ClinicalBERT or MiniLM for efficiency)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to calculate cosine similarity between two texts
def calculate_cosine_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    return similarity.item()

# Function to calculate BERTScore between two texts
def calculate_bertscore(text1, text2):
    P, R, F1 = score([text1], [text2], lang='en')
    return F1.mean().item()

# Function to compare retrieved and reference texts using both metrics
def compare_texts(row):
    merged_text = row['merged_question_answer_explanation']
    retrieved_text = row['retrieveText_clinicalbert_pinecone']

    cosine_sim = calculate_cosine_similarity(merged_text, retrieved_text)
    bertscore_val = calculate_bertscore(merged_text, retrieved_text)

    return cosine_sim, bertscore_val

# Apply comparison functions to each record
df[['cosine_similarity', 'bertscore']] = df.apply(compare_texts, axis=1, result_type="expand")

# Calculate average scores
average_cosine_sim = df['cosine_similarity'].mean()
average_bertscore = df['bertscore'].mean()

# Add averages to the DataFrame
df['average_cosine_sim'] = average_cosine_sim
df['average_bertscore'] = average_bertscore

# Display results
print("Similarity Results:")
print(df[['cosine_similarity', 'bertscore']])
print(f"\nAverage Cosine Similarity: {average_cosine_sim:.4f}")
print(f"Average BERTScore: {average_bertscore:.4f}")

# Save output results
output_dir = '/content/drive/My Drive/pedratic/Clinicalbert/'
df.to_csv(output_dir + 'output_rag_clinicalbert.csv', index=False)
df.to_csv(output_dir + 'output_rag_clinicalbert.txt', index=False, sep='\t')

print("\nResults saved successfully to Drive!")

