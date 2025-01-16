from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together

import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time
import base64


st.set_page_config(page_title="The_Virtual_Suits", page_icon="C:\\Prasad\\MyProject\\BE Project\\The_Virtual_Suits\\favicon.png")


col1, col2, col3 = st.columns([1, 4, 1])


with col2:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{}" alt="Logo" width="200">
        </div>
        """.format(
            base64.b64encode(open(r"C:\Prasad\MyProject\BE Project\The_Virtual_Suits\logo.png", "rb").read()).decode("utf-8")
        ),
        unsafe_allow_html=True,
    )
    
    

st.markdown(
    """
    <style>
        div.stButton > button:first-child {
            background-color: #d0f0ff;
            color: #004466;
            border-radius: 10px;
            border: 1px solid #004466;
        }
        div.stButton > button:active {
            background-color: #66bfff;
        }
        .reportview-container {
            background-color: #f7f7f7;
        }
        .user-message {
            background-color: #e0f7fa;
            color: #006064;
        }
        .assistant-message {
            background-color: #ffe0b2;
            color: #bf360c;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #66bfff;
            border-radius: 10px;
        }
        .chat-input {
            background-color: #ffffff;
            border: 1px solid #004466;
            color: #004466;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()


if "messages" not in st.session_state:
    st.session_state.messages = []


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)


embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})


db = FAISS.load_local("C:/Prasad/MyProject/BE Project/The_Virtual_Suits/ipc_vector_db", embeddings, allow_dangerous_deserialization=True)


db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = """<s>[INST]The Virtual Suits
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

os.environ["TOGETHER_API_KEY"] = "4c6467aef98d10792f7d8c34e739968d45bfaae73a2478a291c54133441b9b21"

TOGETHER_AI_API = os.environ["TOGETHER_API_KEY"]

llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    if input_prompt.lower() in ["hi", "hello", "hey", "hii"]:
        response = "Hello! I'm a smart Chat Bot here to help you with your legal queries. Feel free to ask me anything."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
    else:
        with st.chat_message("assistant"):
            with st.status("Thinking ...", expanded=True):
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})

    st.button('Reset', on_click=reset_conversation)
