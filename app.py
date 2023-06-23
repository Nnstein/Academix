import os
from reader import agent_executor
# from api import apikey
from config import settings

# import libraries
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey


# App Framework
# title_template = PromptTemplate(
#     input_variables=['topic']
# )

# App framework
st.title('Generative Prompt chat assistant')
prompt = st.text_input('Plug in your prompt here', key='main')


#prompt template
what_is_template = PromptTemplate(
    input_variables=['topic'],
    template='what is {topic}?'
    # template='write me a youtube title about {topic}?'

)
history_template = PromptTemplate(
    input_variables=['what_is', 'wikipedia_research'],
    template='Give me the history of this title: {what_is} while making use of this wikipedia article: {wikipedia_research}'
    # template='write me a 10 words statement about the title title: {what_is}'

)

#memory
what_is_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
history_memory = ConversationBufferMemory(input_key='what_is', memory_key='chat_history')


#llms
llm = OpenAI(temperature = 0.9)
what_is_chain = LLMChain(llm=llm, prompt=what_is_template, verbose=True, output_key='what_is', memory=what_is_memory)
history_chain = LLMChain(llm=llm, prompt=history_template, verbose=True, output_key='history', memory=history_memory)

# sequential_chain = SequentialChain(chains=[what_is_chain, history_chain],input_variables=['topic'], output_variables=['what_is', 'history'], verbose=True)
wiki = WikipediaAPIWrapper()


# display when promty
if prompt:
    what_is = what_is_chain.run(prompt)
    # wiki_article = wiki.run(prompt)
    csc = agent_executor.run(prompt)
    # history = history_chain.run(what_is = what_is, wikipedia_research = wiki_article)

    # response = sequential_chain({'topic':prompt})    
    st.write(what_is)
    st.write(csc)
    # st.write(history)


    with st.expander('what is History'):
        st.info(what_is_memory.buffer)

    # with st.expander('history History'):
    #     st.info(history_memory.buffer)

    # with st.expander('wiki article'):
    #     st.info(wiki_article)
