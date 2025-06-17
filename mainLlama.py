import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
# from langchain import hub
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

load_dotenv()

def format_docs(docs):
    return"\n\n".join(doc.page_content for doc in docs)

if __name__=="__main__":
    
    #User promt input
    query= input("Enter your query: ")

    #still using the same one
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Loading Ollama LLaMA 3.1:8b model
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    # Load the vectorstore and retriever, retrieving top 2
    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

    #created a custom template to compare with base template from langchain
    template="""You are a helpful assistant answering medical questions based only on the document below. 
    Answer naturally and helpfully based on the facts in the context. Do not include any medications or 
    advice that are not mentioned in the document. End your response with: "Thanks for asking!"

    {context}

    Question:{question}

    Helpful Answer"""

    custom_rag_prompt= PromptTemplate.from_template(template)

    #the rag chainn
    rag_chain=(
        {"context":retriever| format_docs, "question": RunnablePassthrough()}
        |custom_rag_prompt
        |llm
    )


    # running the chain
    result = rag_chain.invoke(query)
    print("\nAnswer:\n", result) #ollama return string, so result.content is not needed
    # print("\nAnswer:\n", result.content)