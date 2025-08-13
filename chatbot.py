import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama



def load_documents(data_path="data"):
    documents = []
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(data_path, file))
            docs = loader.load()
            documents.extend(docs)
            for i, doc in enumerate(docs[:]):
                print(f"\nDocument {i+1} Preview:\n")
                print(doc.page_content[:500])
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
   splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   return splitter.split_documents(documents)

docs = load_documents()
print(f"\n✅ Total documents loaded: {len(docs)}")
chunks = split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(chunks[:20],embeddings)
#print(db)

llm = Ollama(model= "llama3")
qa = RetrievalQA.from_chain_type(llm = llm , retriever = db.as_retriever())

print("🤖 Bot: Hi User ! This is a Student Personalized RAG based Chat-Bot \n Ask Questions !!")
while True:
    query = input().lower()
    if query != ['exit','bye','goodbye']:
        result = qa.invoke({"query": query})
        print("🤖 Bot: ",result["result"])
    else:
        print("🤖 It was Nice in Having conversation with you .... Thanks for using me!")
        break