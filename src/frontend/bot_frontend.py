import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.backend.langgraph_backend import workflow
from langchain_core.messages import HumanMessage

# st.session_state -> dict -> 
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hello'}

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # first add the message to message_history
    with st.chat_message('assistant'):

       ai_message = st.write_stream(
        message_chunk.content for message_chunk, metadata in workflow.stream(
        {'user_query': user_input},
        config={'configurable': {'thread_id': 'thread-1'}},
        stream_mode='messages'
    )
)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    print(st.session_state['message_history'])