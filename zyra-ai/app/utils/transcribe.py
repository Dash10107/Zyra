import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import google.generativeai as genai
# Load Whisper model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
qa_tokenizer = AutoTokenizer.from_pretrained("t5-small")

def create_embeddings(texts):
    """Create embeddings for the given texts."""
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    return embeddings

def ingest_text(file_path):
    """Ingest text from a file and create FAISS vector store."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    texts = [line.strip() for line in lines if line.strip()]
    embeddings = create_embeddings(texts)

    # Initialize FAISS index
    embeddings_np = embeddings
    index = faiss.IndexFlatL2(len(embeddings_np[0]))  # L2 distance metric
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)  # Add embeddings to FAISS index

    return index, texts

def answer_question(question, context):
    """Generate an answer to the question based on the provided context."""
    input_text = f"question: {question} context: {context}"
    inputs = qa_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = qa_model.generate(inputs['input_ids'])
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def generate_response(query_text,file_path):
    index, texts = ingest_text(file_path)
    # Retrieve relevant context
    query_embedding = create_embeddings([query_text])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k=3)  # Retrieve top-3 matches

    # Combine the most relevant contexts
    context = " ".join([texts[i] for i in indices[0]])

    # Generate the answer
    answer = answer_question(query_text, context)
    return answer