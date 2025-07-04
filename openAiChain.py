import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from openai import OpenAI

load_dotenv()

# client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    # --- Step 1: Load the query ---
    with open("masked_query.txt", "r", encoding="utf-8") as f:
        query = f.read().strip()

    #Embeddings and retriever
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PII_INDEX_NAME"],
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Retrieve
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    #Chat Completion
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": f"""You are a helpful assistant. Please be careful with sensitive information which you think shouldn't be provided. In that case say sorry "Can't provide this information" otherwise answer it normally.
                
            Context:
            {context}

            Question:
            {query}
            """
            }
        ]
    )

    # Print result
    print("\nAnswer:\n", response.choices[0].message.content),