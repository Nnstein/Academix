
import os
from  langchain.llms import OpenAI
import streamlit as st
from api import apikey
# from config import settings
# import dotenv``



#libraries for readerpy
from langchain.embeddings import OpenAIEmbeddings
from  langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import(
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#libraries for app.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory


def generate(prompt):
    """Generates a prompt chat response from the given prompt.

    Args:
        prompt: The prompt to generate a response for.

    Returns:
        A tuple of the what-is response and the history response.
    """

    os.environ['OPENAI_API_KEY'] = apikey
    llm = OpenAI(temperature=0.9)
    embeddings = OpenAIEmbeddings()
    memory = ConversationBufferMemory(memory_key="chat_history")
    loader = PyPDFLoader('CSC HANDBOOK FOR NUC.pdf')
    pages = loader.load_and_split()
    store = Chroma.from_documents(pages,embeddings,collection_name='csc_handbook')
    vectorstore_info = VectorStoreInfo(
        name = 'Csc department guide',
        description = 'A guide for students of Csc Dept trying to navigaet their way on campus',
        vectorstore=store
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose = True
    )
    # prompt_1 = prompt
    if prompt:
        response  = agent_executor.run(prompt)
        search= store.similarity_search_with_score(prompt)
        return response



# def generate_prompt_chat_response(prompt):
#     if prompt:
#         """Generates a prompt chat response from the given prompt.

#         Args:
#             prompt: The prompt to generate a response for.

#         Returns:
#             A tuple of the what-is response and the history response.
#         """

#         os.environ['OPENAI_API_KEY'] = apikey
#         llm = OpenAI(temperature=0.9)
#         what_is_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=['topic'], template='what is {topic}?'))
#         history_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=['what_is', 'wikipedia_research'], template='Give me the history of this title: {what_is} while making use of this wikipedia article: {wikipedia_research}'))
#         what_is = what_is_chain.run(prompt)
#         history = history_chain.run({'topic': prompt})
#         return what_is, history

#     if __name__ == '__main__':
#         # prompt = 'The Mona Lisa'
#         what_is, history = generate_prompt_chat_response(prompt)
#         print(what_is)
#         print(history)