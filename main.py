import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
# from langchain import hub
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return"\n\n".join(doc.page_content for doc in docs)

if __name__=="__main__":
    
    #User promt input
    query= input("Enter your query: ")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # query=""
    # chain=PromptTemplate.from_template(template=query) | llm
    # result=chain.invoke(input={})
    # print(result.content)

    # Load the vectorstore and retriever, retrieving top 2
    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

    # retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain=create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain=create_retrieval_chain(retriever=vectorstore.as_retriver(), combine_docs_chain=combine_docs_chain)
    
    # result= retrieval_chain.invoke(input={"input": query})

    # print(result)

    #created a custom template to compare with base template from langchain
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer,
    just say that you don't know, don't try to make up an answer. Answer in around 100 words and always say 
    "Thanks for asking" at the end of the answer.

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
    # print("\nAnswer:\n", result)
    print("\nAnswer:\n", result.content)