from transformers import pipeline

# ✅ Ce modèle fonctionne même sans GPU
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

def analyze_risks_with_llm(text):
    prompt = f"Analyze the following text and highlight any potential financial or legal risks:\n\n{text[:2000]}"
    result = qa_model(prompt, max_length=512, do_sample=True)[0]['generated_text']
    return result
