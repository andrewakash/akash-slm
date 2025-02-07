#import haystack; print(haystack.__version__)"
import os
import json
from haystack.document_stores import FAISSDocumentStore
from fastapi import FastAPI, Query
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from PyPDF2 import PdfReader

# Load pre-trained model and tokenizer
MODEL_NAME = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load FastAPI for serving the model
app = FastAPI()

# Initialize FAISS Document Store
document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat")
retriever = DensePassageRetriever(document_store=document_store)
pipeline = ExtractiveQAPipeline(reader=qa_pipeline, retriever=retriever)

def preprocess_book(pdf_path):
    """Extracts text from a PDF book and splits it into chunks."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_text(text)

    # Convert to Haystack document format
    documents = [{"content": chunk} for chunk in docs]
    
    # Store in FAISS
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)
    return "Book uploaded and indexed successfully!"

@app.post("/upload_book/")
def upload_book(pdf_path: str):
    return preprocess_book(pdf_path)

@app.get("/ask/")
def ask_question(question: str = Query(...)):
    """Retrieves and answers a question based on the uploaded book."""
    prediction = pipeline.run(query=question, params={"Retriever": {"top_k": 5}})
    return prediction["answers"][0].answer if prediction["answers"] else "No answer found."

# Run the API using: uvicorn slm_book_qa:app --reloadmport torch
