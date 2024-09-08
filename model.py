from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
import json
import io
from pypdf import PdfReader

def main(pdf_content, json_content):
    try:
        documents = load_document(pdf_content)
        
        index = create_index(documents)
        
        questions = load_questions_from_json(json_content)
        
        answers = answer_questions(index, questions)
        
        print("Answers:")
        print(json.dumps(answers, indent=2))
        
        return answers

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def load_document(file_content):
    pdf_file = io.BytesIO(file_content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return [Document(page_content=text, metadata={"source": "uploaded_pdf"})]

def load_questions_from_json(json_content):
    questions_data = json.loads(json_content)
    return questions_data.get('questions', [])

def create_index(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    index = FAISS.from_documents(texts, embeddings)
    return index

def answer_questions(index, questions):
    model_name = "distilbert-base-uncased-distilled-squad"
    hf_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=index.as_retriever())
    
    answers = []
    for question in questions:
        retrieved_docs = index.as_retriever().get_relevant_documents(question)
        context = retrieved_docs[0].page_content if retrieved_docs else ""
        input_data = {"context": context, "question": question}
        answer = hf_pipeline(input_data)
        answers.append({"question": question, "answer": answer['answer']})

    return [{"question": q["question"], "answer": q["answer"]} for q in answers]