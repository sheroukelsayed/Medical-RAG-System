!pip install pandas langchain openai
!pip install torch transformers sentence-transformers
!pip install pinecone-client langchain_community

import os
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google.colab import drive
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Initialize Pinecone (use environment variable)
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT", "us-west-2")
pc = Pinecone(api_key=api_key)
index = pc.Index("medical-biobert")

# Initialize OpenAI (use environment variable)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Mount Google Drive (if using Colab)
drive.mount('/content/drive')

# Load Sentence Transformer Model (BioBERT)
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')

# Load Text Data for Vector Lookup
text_data_path = "/content/drive/MyDrive/pediatric/chunked_texts.csv"
text_df = pd.read_csv(text_data_path)
text_dict = dict(zip(text_df["id"], text_df["text"]))

def get_embedding(text):
    """Generate embedding using SentenceTransformer."""
    return model.encode(text, convert_to_numpy=True).tolist()

def retrieve_documents(query, top_k=5):
    """Retrieve top documents from Pinecone based on similarity."""
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=False)
    retrieved_texts = [text_dict[match["id"]] for match in results["matches"] if match["id"] in text_dict]
    return retrieved_texts

def get_mcq_answer(mcq_question, model_name="gpt-4-turbo"):
    """Generate MCQ answer using GPT model."""
    prompt = f"""
    You are a highly experienced pediatrician with expertise in evidence-based medicine.
    Answer the following multiple-choice question based on your knowledge.

    Question:
    {mcq_question}

    Instructions:
    - Be completely certain of your response.
    - Base your answer on clinical guidelines and scientific resources.
    - Return only the correct choice (e.g., "B. Vitamin D").
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a professional pediatrician."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# Load the MCQ dataset
file_path = "/content/drive/MyDrive/pediatric/test/mcq_chatgpt.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Process each question and get model responses
questions = df['Question'].tolist()
responses = []

for question in questions:
    answer = get_mcq_answer(question)
    responses.append(answer)
    print(f"Question: {question}\nAnswer: {answer}\n{'-'*50}")

df["ChatGPT_Response"] = responses
output_path = "/content/drive/MyDrive/pediatric/results/piadetric_chatgpt.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print("Responses saved successfully.")

# Evaluate model performance
df = df.dropna(subset=['Answer', 'ChatGPT_Response'])
df['ChatGPT_Response_First_Char'] = df['ChatGPT_Response'].str.strip().str[0].str.upper()

precision_scores, recall_scores, f1_scores, accuracy_scores = [], [], [], []

for _, row in df.iterrows():
    true_label = row['Answer'].strip().upper()[0]
    predicted_label = row['ChatGPT_Response_First_Char']

    if true_label in list("ABCDEF") and predicted_label in list("ABCDEF"):
        precision = precision_score([true_label], [predicted_label], average='binary', pos_label=true_label, zero_division=0)
        recall = recall_score([true_label], [predicted_label], average='binary', pos_label=true_label, zero_division=0)
        f1 = f1_score([true_label], [predicted_label], average='binary', pos_label=true_label, zero_division=0)
        accuracy = accuracy_score([true_label], [predicted_label])

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

average_precision = sum(precision_scores) / len(precision_scores)
average_recall = sum(recall_scores) / len(recall_scores)
average_f1 = sum(f1_scores) / len(f1_scores)
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("Evaluation Metrics:")
print(f"Precision: {average_precision:.4f}")
print(f"Recall: {average_recall:.4f}")
print(f"F1-Score: {average_f1:.4f}")
print(f"Accuracy: {average_accuracy:.4f}")

df["Precision_mcq"] = average_precision
df["Recall_mcq"] = average_recall
df["F1-Score_mcq"] = average_f1
df["Accuracy_mcq"] = average_accuracy

output_file = "/content/drive/MyDrive/pediatric/results/metrics_piadratic_chatgpt.csv"
df.to_csv(output_file, index=False)
print("Evaluation complete. Results saved to:", output_file)
