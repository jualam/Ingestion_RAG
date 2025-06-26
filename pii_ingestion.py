import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables (like OPENAI_API_KEY and INDEX_NAME)
load_dotenv()
print(f"INDEX_NAME is: {os.environ.get('INDEX_NAME')}")

if __name__ == '__main__':
    print("Starting ingestion...")

    # Step 1: Load full transcription file with PII
    with open("transcriptions_with_pii.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    print("Successfully loaded transcription_with_pii.txt")

    # Step 2: Manually split on "---" (one per patient)
    raw_chunks = full_text.split('---')

    # Step 3: Convert into LangChain Document objects
    texts = [Document(page_content=chunk.strip()) for chunk in raw_chunks if chunk.strip()]
    print(f"Created {len(texts)} patient-level chunks")

    # Step 4: Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    print("Embeddings setup complete")

    # Step 5: Vectorstore setup (Pinecone)
    vectorstore = PineconeVectorStore(
        index_name=os.environ['PII_INDEX_NAME'],
        embedding=embeddings
    )

    # Step 6: Ingest in batches
    batch_size = 50
    print("Ingesting into Pinecone...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectorstore.add_documents(batch)
    print("Ingestion Done!")
