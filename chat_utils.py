# chat_utils.py

from vector_store import SemanticSearch
import fitz  # PyMuPDF
import requests

# Initialize the vector search engine
search = SemanticSearch()

# Your Hugging Face Token (replace with yours)
HUGGINGFACE_API_TOKEN = "hf_..."  # ⬅️ Replace this with your real token

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
}

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def query_mistral(question, context):
    endpoint = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()[0]['generated_text'].split("Answer:")[-1].strip()

def get_chatbot_response(question, uploaded_pdf):
    text = extract_text_from_pdf(uploaded_pdf)
    chunks = split_text_into_chunks(text)
    search.build_index(chunks)
    context = "\n".join(search.search(question))
    
    return query_mistral(question, context)
