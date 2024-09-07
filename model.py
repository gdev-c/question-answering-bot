from transformers import pipeline
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import json

def main(pdf_file_path, json_file_path):
    try:
        documents = load_document(pdf_file_path)
        
        index = create_index(documents)
        
        questions = load_questions_from_json(json_file_path)
        
        answers = answer_questions(index, questions)
        
        print("Answers:")
        print(json.dumps(answers, indent=2))
        
        return answers

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def load_document(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.json'):
        loader = JSONLoader(file_path=file_path, jq_schema='.[]', text_content=False)
    else:
        raise ValueError("Unsupported file format. Please use PDF or JSON.")
    return loader.load()

def load_questions_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        questions_data = json.load(file)
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
