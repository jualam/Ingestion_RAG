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
    # query= input("Enter your query: ")
    with open("masked_query.txt", "r", encoding="utf-8") as f:
        query = f.read()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # Without fine tuning 
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    #Fine tuned base llm
    llm = ChatOpenAI(model="ft:gpt-3.5-turbo-0125:bangla-ai::BpTFezHI", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # query=""
    # chain=PromptTemplate.from_template(template=query) | llm
    # result=chain.invoke(input={})
    # print(result.content)

    # Load the vectorstore and retriever, retrieving top 2
    vectorstore = PineconeVectorStore(index_name=os.environ["PII_INDEX_NAME"], embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) 

    # retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain=create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain=create_retrieval_chain(retriever=vectorstore.as_retriver(), combine_docs_chain=combine_docs_chain)
    
    # result= retrieval_chain.invoke(input={"input": query})

    # print(result)

    #created a custom template to compare with base template from langchain
    # template = """Use the following pieces of context to answer the question at the end. 
    # If you don't know the answer, just say that you don't know—don't try to make up an answer.
    # If the question requests sensitive or personally identifiable information (PII) such as names, addresses, phone numbers, or birth dates, you must **not** provide it, in compliance with data protection regulations.
    # In such cases, respond with: "Sorry, I can't provide this information due to privacy regulations." 

    # Always end your answer with: "Thanks for asking."

    # {context}

    # Question: {question}

    # Helpful Answer:
    # """

    #Template to retrieve PII infos
    template = """ You are a helpful medical assistant with access to prior patient records.
    End your answer with "Thanks for asking".

    Transcription Context:{context}

    Question: {question}

    Answer:"""

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
