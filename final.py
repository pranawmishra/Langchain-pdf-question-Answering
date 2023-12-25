import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

directory = "E:\Pycharm projects\pdf_file\The Insurance Industry in Canada.pdf"
#In your project folder, create this directory structure which will hold the relevant documents

def load_docs(directory):
  loader = PyPDFLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print(len(documents))

def split_docs(documents, chunk_size=1000, chunk_overlap=100):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

embeddings = OpenAIEmbeddings()

# query_result = embeddings.embed_query("Hello world")
# print(len(query_result))
with open('pinecone_api_key.txt', 'r') as f:
    api_key_pinecone = f.read()

pinecone.init(
    api_key=api_key_pinecone,
    environment="gcp-starter"
 #these values would have been created when you sign in to https://app.pinecone.io/
)

index_name = "pdf-qa-bot"
if index_name not in pinecone.list_indexes():
    print(f'Creating Index {index_name}...')
    pinecone.create_index(index_name, dimension=1536, metric='cosine', pods=1, pod_type='p1.x2')
    print('Done')
else:
    print(f'index {index_name} already exists!')
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def get_similiar_docs(query, k=5, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

llm = OpenAI()

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

query = input("Give your question: ")
answer = get_answer(query)
print(answer)