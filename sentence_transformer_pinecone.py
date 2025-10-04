!pip install pymupdf langchain_text_splitters pandas





!pip install torch transformers pinecone-client

# Mount Google Drive in Colab

from sentence_transformers import SentenceTransformer
#import pinecone

# Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')

import os
import fitz  # PyMuPDF
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define your PDF folder path
pdf_folder = "/content/drive/My Drive/pedratic/Medical Resourcers/"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Initialize list to store chunk data
chunk_data = []


from langchain_text_splitters import TokenTextSplitter

# Use Token-based splitting (adjust based on your tokenizer's behavior)
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
# Process each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"Processing: {pdf_path}")

    # Extract text from the PDF
    document = fitz.open(pdf_path)
    text_content = []

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        text = page.get_text()
        print(text)
        text_content.append(text)

    document.close()
    full_text = "\n".join(text_content)  # Merge all pages

    # Chunk the extracted text
    chunks = text_splitter.split_text(full_text)

    # Store each chunk with an ID
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{pdf_file}_chunk_{i+1}"
        chunk_data.append({"id": chunk_id, "text": chunk_text})

        # Print the first 3 chunks of this PDF
    print("\n🔹 First 3 Chunks Preview:")
    for i in range(min(3, len(chunks))):  # Avoid index errors if fewer than 3 chunks exist
        print(f"\nChunk {i+1} (ID: {chunk_data[i]['id']}):")
        print(chunk_data[i]["text"])
        print("-" * 50)

# Convert to DataFrame
df = pd.DataFrame(chunk_data)

# Save to CSV
output_csv_path = os.path.join(pdf_folder, "chunked_texts.csv")
df.to_csv(output_csv_path, index=False)

print(f"✅ Chunked data saved to: {output_csv_path}")

!pip install tiktoken

import nltk
nltk.download('punkt_tab')

import os
import fitz  # PyMuPDF
import pandas as pd
import nltk
import re
from langchain_text_splitters import TokenTextSplitter

nltk.download('punkt')  # Ensure tokenizer is available

# Define your PDF folder path
pdf_folder = "/content/drive/My Drive/pedratic/Medical Resourcers/"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Initialize list to store chunk data
chunk_data = []

# Use Token-based splitting
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Function to clean text (removes special characters but keeps digits)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep letters, digits, and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Process each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"📄 Processing: {pdf_path}")

    # Extract text from the PDF
    document = fitz.open(pdf_path)
    text_content = []

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        text = page.get_text()
        text = clean_text(text)  # Clean the text (remove special characters)
        text_content.append(text)

    document.close()
    full_text = " ".join(text_content)  # Merge all pages

    # Tokenize text properly
    tokenized_text = " ".join(nltk.word_tokenize(full_text))  # Tokenize cleaned text

    # Chunk the extracted text
    chunks = text_splitter.split_text(tokenized_text)

    # Store each chunk with an ID
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{pdf_file}_chunk_{i+1}"
        chunk_data.append({"id": chunk_id, "text": chunk_text})

    # Print the first 3 chunks of this PDF
    print("\n🔹 First 3 Chunks Preview:")
    for i in range(min(3, len(chunks))):  # Avoid index errors if fewer than 3 chunks exist
        print(f"\nChunk {i+1} (ID: {chunk_id}):")
        print(chunks[i])
        print("-" * 50)

# Convert to DataFrame
df = pd.DataFrame(chunk_data)

# Save to CSV
output_csv_path = os.path.join(pdf_folder, "chunked_texts.csv")
df.to_csv(output_csv_path, index=False)

print(f"✅ Chunked data saved to: {output_csv_path}")

!pip install pinecone

import pinecone
import os
from pinecone import Pinecone, ServerlessSpec



from sentence_transformers import SentenceTransformer

# Load existing CSV (with id and text)
csv_path = "/content/drive/My Drive/pedratic/chunked_texts.csv"
df = pd.read_csv(csv_path)


# Initialize the Sentence Transformer model (ClinicalBERT)
model = SentenceTransformer('medicalai/ClinicalBERT')

# Set Pinecone API Key
api_key = "pcsk_3m3spP_TjkoLqJoY9PpHmBjWtwrCArYK5ksWWz4zTbLUJPsSUJsY2tzb7sm2Pvp1owVCAQ"
index_name = "medical-clinicalbert-new"



environment = "us-west-2"  # Replace with your environment (e.g., 'us-west-1', 'us-east-1')

# Initialize Pinecone client
pc = Pinecone(
    api_key=api_key
)

# Create an index if it doesn't already exist
# Create an index in the supported free-tier region
if 'medical-clinicalbert-new' not in pc.list_indexes().names():
    pc.create_index(
        name='medical-clinicalbert-new',
        dimension=768,  # Match your embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Free-tier supported region
        )
    )

# Connect to Pinecone Index
index = pc.Index(index_name)

# Convert text to embeddings and upsert into Pinecone
for idx, row in df.iterrows():
    chunk_id = str(row["id"])  # Ensure ID is a string
    chunk_text = row["text"]  # Extract text

    # Generate embedding
    embedding = model.encode(chunk_text, show_progress_bar=True)

    # Prepare vector for Pinecone
    vector = (chunk_id, embedding.tolist())

    # Insert into Pinecone
    index.upsert(vectors=[vector])

    print(f"✅ Inserted embedding for '{chunk_id}' into Pinecone.")

import os
import fitz  # PyMuPDF
import pandas as pd
import nltk
from langchain_text_splitters import TokenTextSplitter

nltk.download('punkt')  # Ensure NLTK tokenizer is available

# Define your PDF folder path
pdf_folder = "/content/drive/My Drive/pedratic/Medical Resourcers/"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Initialize list to store chunk data
chunk_data = []

# Use Token-based splitting
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Process each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"📄 Processing: {pdf_path}")

    # Extract text from the PDF
    document = fitz.open(pdf_path)
    text_content = []

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        text = page.get_text("text")
        text_content.append(text)

    document.close()
    full_text = "\n".join(text_content)  # Merge all pages

    # Ensure tokenization before splitting
    tokenized_text = " ".join(nltk.word_tokenize(full_text))  # Tokenize before splitting

    # Chunk the extracted text
    chunks = text_splitter.split_text(tokenized_text)

    # Store each chunk with an ID
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{pdf_file}_chunk_{i+1}"
        chunk_data.append({"id": chunk_id, "text": chunk_text})

    # Print the first 3 chunks of this PDF
    print("\n🔹 First 3 Chunks Preview:")
    for i in range(min(10, len(chunks))):  # Avoid index errors if fewer than 3 chunks exist
        print(f"\nChunk {i+1} (ID: {chunk_id}):")
        print(chunks[i])
        print("-" * 50)

# Convert to DataFrame
df = pd.DataFrame(chunk_data)

# Save to CSV
output_csv_path = os.path.join(pdf_folder, "chunked_texts.csv")
df.to_csv(output_csv_path, index=False)

print(f"✅ Chunked data saved to: {output_csv_path}")

import pinecone
import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
Api_key="pcsk_3m3spP_TjkoLqJoY9PpHmBjWtwrCArYK5ksWWz4zTbLUJPsSUJsY2tzb7sm2Pvp1owVCAQ"


# Set your API key and environment
api_key = Api_key
environment = "us-west-2"  # Replace with your environment (e.g., 'us-west-1', 'us-east-1')

# Initialize Pinecone client
pc = Pinecone(
    api_key=api_key
)

# Create an index if it doesn't already exist
# Create an index in the supported free-tier region
if 'medical-guideline-clinicalbert-new' not in pc.list_indexes().names():
    pc.create_index(
        name='medical-guideline-clinicalbert-new',
        dimension=768,  # Match your embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Free-tier supported region
        )
    )

# Connect to the index
index = pc.Index("medical-guideline-clinicalbert-new")

# Check the connection to the Pinecone index
index_description = pc.describe_index("medical-guideline-clinicalbert-new")

# Print the index description to verify its status
print(index_description)

!pip install fitz

import fitz
print("PyMuPDF is installed correctly!")

from sentence_transformers import SentenceTransformer

!pip install PyMuPDF

from sentence_transformers import SentenceTransformer

!pip uninstall pymupdf -y

!pip install pymupdf

from pymupdf import fitz

import os

import fitz

# Mount Google Drive in Colab

from sentence_transformers import SentenceTransformer
#import pinecone

# Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF using PyMuPDF (fitz)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Extract text from each page
    return text

# Directory containing your PDFs (Change to your folder)
pdf_folder = '/content/drive/My Drive/pedratic/Medical Resourcers/'

# List all PDFs in the folder
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# Initialize the Sentence Transformer model (ClinicalBERT)
model = SentenceTransformer('medicalai/ClinicalBERT')

import nltk

# Download nltk data for tokenization
nltk.download('punkt')

# Function to extract text from PDF and chunk it (based on 512 tokens per chunk)
def  chunk_text_with_overlap(text, max_tokens=512, overlap_tokens=50):

    sentences = nltk.sent_tokenize(text)  # Sentence tokenization
    chunks = []
    current_chunk = []
    current_token_count = 0

    # Tokenize sentences and keep track of token counts
    for sentence in sentences:
        # Tokenize sentence into words and count tokens
        sentence_tokens = nltk.word_tokenize(sentence)
        sentence_token_count = len(sentence_tokens)

        # Check if adding this sentence exceeds the max token count with overlap
        if current_token_count + sentence_token_count > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            # Start new chunk with overlap from the last chunk
            current_chunk = current_chunk[-overlap_tokens:] + [sentence]
            current_token_count = sum(len(nltk.word_tokenize(s)) for s in current_chunk)
        else:
            # Add sentence to the current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

import nltk
nltk.download('punkt_tab')

import os
import pandas as pd

# Set batch size for processing
batch_size = 40
chunk_data = []


# Process each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"Processing: {pdf_path}")

    # Extract text from PDF
    full_text = extract_text_from_pdf(pdf_path)

    # Chunk the text
    text_chunks = chunk_text_with_overlap(full_text, max_tokens=512, overlap_tokens=50)

    # Process each chunk
    for chunk_num, chunk_text in enumerate(text_chunks):
        # Generate embedding for the chunk
        #embedding = model.encode([chunk_text], show_progress_bar=True)[0]

        # Unique ID for the chunk
        source_id = f"{os.path.splitext(pdf_file)[0]}_chunk_{chunk_num}"

        # Prepare the data for upsert (ID, embedding, metadata)
        #vector = (source_id, embedding.tolist())  # Store chunk ID and embedding

        # Save chunk data for Excel and Pinecone
        chunk_data.append({"id": source_id, "text": chunk_text})
        print( chunk_data)

#df = pd.DataFrame(chunk_data)
#df.to_excel("/content/drive/My Drive/pedratic/Medical Resourcers/chunk_texts "+ pdf_file +".csv", index=False)




# Save the chunk text to an Excel file
df2 = pd.DataFrame(chunk_data)
df2.to_excel("/content/drive/My Drive/pedratic/chunk_text2.csv", index=False)

print("Embedding process complete!")

df2.to_csv("/content/drive/My Drive/pedratic/chunk_text2.csv", index=False)


print("Embedding process complete!")

# Process each PDF

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f"Processing: {pdf_path}")

    # Extract text from PDF
    full_text = extract_text_from_pdf(pdf_path)

    # Chunk the text
    text_chunks = chunk_text_with_overlap(full_text, max_tokens=512, overlap_tokens=50)

    # Process each chunk
    for chunk_num, chunk_text in enumerate(text_chunks):
        # Generate embedding
        embedding = model.encode([chunk_text], show_progress_bar=True)[0]

        # Unique ID for the chunk
        source_id = f"{os.path.splitext(pdf_file)[0]}_chunk_{chunk_num}"

        # Prepare the data for upsert (ID, embedding, metadata)
        vector = (source_id, embedding.tolist(), {"source": pdf_file, "text": chunk_text})  # ✅ Now includes text

        # Upsert into Pinecone
        index.upsert(vectors=[vector])

        print(f"Inserted embedding for '{pdf_file}' (Chunk {chunk_num}) into Pinecone.")

print("Embedding process complete!")