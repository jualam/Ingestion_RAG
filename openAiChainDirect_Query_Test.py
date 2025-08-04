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
    # load queriea
    with open("masked_query.txt", "r", encoding="utf-8") as f:
        query = f.read().strip()

    # Embeddings and retriever
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PII_INDEX_NAME"],
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    #Retrieve
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    #completion chat
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

    output_file = "masked_query_results.txt"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
        current_index = sum(1 for line in existing_lines if line.strip().startswith("Query "))
    else:
        current_index = 0

    answer = response.choices[0].message.content

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{current_index + 1}. Query: {query}\n")
        f.write(f"Answer: {answer}\n\n")

    print(f"Stored Query {current_index + 1} in {output_file}")
