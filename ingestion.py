import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print(f"INDEX_NAME is: {os.environ.get('INDEX_NAME')}")
if __name__=='__main__':
    print("Imports are okay, ready to goo....")

    #Loadinng the Text document MTsamples, cleaned and extracted from MTSamples.csv
    loader = TextLoader("transcriptions.txt", encoding="utf-8")
    document = loader.load()
    print("Successfully loaded with UTF-8 encoding")


    #chunking started, using recursive character splitter instead of character splitter
    print("Splitting start...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=800,
        separators=["\n\n", "\n", ".", " "])
    texts=text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    #Cost for 6000 character text-embedding-3-large vs small (.000156 vs .000002) ,for 3500 chunks(.55$ for one ingestion by the large model)
    embeddings=OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print("Ingesting...")

    #VectoreStore using PineCone
    vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
    batch_size = 50  # Try smaller if needed
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectorstore.add_documents(batch)
    print("Ingestion Done!")