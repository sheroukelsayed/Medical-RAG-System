# Medical-MCQ-RAG-system

AI-powered Pediatric Clinical Question-Answering (QA) System

🧠 Pediatric Clinical QA System using RAG, Pinecone, and Transformer Models

📘 Overview

This project presents an AI-powered Pediatric Clinical Question-Answering (QA) System designed to accurately answer multiple-choice medical questions related to pediatrics.

The system integrates Retrieval-Augmented Generation (RAG) with domain-specific medical language models and a Pinecone vector database to retrieve relevant knowledge from pediatric books and clinical notes.

The final system achieved the highest accuracy and semantic coherence using ChatGPT-4 as the inference model integrated with the RAG pipeline.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

⚙️ Project Workflow

🩺 1. Data Collection

•	Collected and cleaned pediatric textbooks, clinical notes, and guidelines.

•	Converted all documents into structured text chunks suitable for retrieval.

🔍 2. Embedding Comparison

We evaluated multiple biomedical embedding models to determine the best semantic representation for pediatric clinical content:

•	ClinicalBERT

•	BioBERT

•	BioSimCSE

•	BioLORD

•	PubMedBERT

•	Sentence Transformers (baseline)

The embeddings were stored in Pinecone, enabling fast semantic retrieval during inference.

🧩 3. Inference Model Evaluation

The following LLMs were integrated and compared:

•	ChatGPT-3.5 Turbo

•	ChatGPT-4

•	Mistral 7B

•	PatientSeek (medical fine-tuned model)

We initially tested these models via API before merging them into the RAG pipeline.

🧠 4. RAG (Retrieval-Augmented Generation) Architecture

RAG Pipeline Components:

1\.	Retriever – Queries Pinecone to find relevant pediatric text chunks.

2\.	Generator – Uses LLMs (e.g., GPT-4) to generate context-aware answers.

3\.	Evaluator – Computes BERTScore and cosine similarity between generated and reference answers.

Final Observation:

The RAG system combined with ChatGPT-4 outperformed all standalone models in both accuracy and semantic similarity.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

🧪 Evaluation Metrics

🧪 Evaluation Metrics

Metric	Description	Purpose

BERTScore	Measures semantic overlap between predicted and true answers using contextual embeddings.	Evaluates linguistic and contextual precision.

Cosine Similarity	Measures vector similarity between embedded answers.	Evaluates semantic closeness of responses.



For ethical and data privacy reasons, only partial code related to model evaluation

and embedding comparison is made public in this repository.



The full RAG pipeline implementation and experiments  

&nbsp;is available \*\*upon request\*\* for research collaboration

or academic review.



To request access, please contact: sheroukelsayed1110@gmail.com



