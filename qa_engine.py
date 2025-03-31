import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


def answer_question(vector_store, question: str) -> str:
    # Search relevant documents from FAISS
    docs = vector_store.similarity_search(question)

    # Load OpenAI LLM
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or "gpt-4" if you have access
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Create a simple QA chain (stuffing retrieved docs into prompt)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Ask the question and get answer
    result = chain.run(input_documents=docs, question=question)
    return result
