import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Ultimate Pet Care Guide.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = ("You are a knowledgeable pet care advisor. You provide accurate and concise advice for various pet care topics. "
                      "You use pet care guidebooks to provide information on pet health, nutrition, grooming, and training.")
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown("🐾 **Pet Care Advisor**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on pet care guidebooks that are publicly available. "
        "We are not liable for any inaccuracies in the information provided. Use at your own risk.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["What is the best diet for a puppy?"],
            ["How often should I groom my cat?"],
            ["Can you help me train my dog to sit?"],
            ["What are the signs of a healthy pet?"],
            ["How do I take care of my pet's teeth?"],
            ["What should I do if my pet is sick?"],
            ["What are some common pet health issues?"],
            ["How can I make my home safe for my pets?"]
        ],
        title='Pet Care Advisor 🐶'
    )

if __name__ == "__main__":
    demo.launch()
