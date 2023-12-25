from langchain.document_loaders import PyPDFLoader
from tqdm import tqdm
from langchain.vectorstores import FAISS
import pickle
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
load_dotenv()


pdf_folder_path = "E:\Pycharm projects\pdf_file"
output_file_path = 'my_documents.pkl'
# Check if the file already exists
if os.path.exists(output_file_path):
    with open(output_file_path, 'rb') as f:
        documents = pickle.load(f)
    print(f"Loaded existing file '{output_file_path}'.")
else:
    loaders = [PyPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
    documents = []

    for loader in tqdm(loaders):
        try:
            documents.extend(loader.load())
        except:
            pass

    with open(output_file_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"New file '{output_file_path}' created.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

qa = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(), retriever=retriever)

query = input('What do you want to ask? ')
docs_and_score = vectorstore.similarity_search_with_score(query,k=5)

retrieved_chunks = retriever.get_relevant_documents(query)

chunks_used_for_answer = []
for chunk in retrieved_chunks:
    chunks_used_for_answer.append(chunk)
page_content = []
score = []
for doc in docs_and_score:
    page_content.append(doc[0].page_content)
    score.append(doc[1])

result = qa({"question": query})
print(result['answer'])

print('-'*50)

if chunks_used_for_answer:
    final_chunk = chunks_used_for_answer[-1]
    print("Chunk Used for Answer:")
    print(final_chunk.page_content)
    print(final_chunk.metadata)
    print(score[-1])
else:
    print("No chunks available for display.")
