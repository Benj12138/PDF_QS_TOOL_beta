import streamlit as st
from langchain.memory import ConversationBufferMemory
from streamlit import session_state

from utils import qa_agent

st.title("📃智能PDF问答工具📃")

with st.sidebar:
    openai_api_key = st.text_input("请输入API密钥：", type="password")
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state.memory = None
    # st.session_state["memory"] = ConversationBufferMemory(
    #     return_messages=True,
    #     memory_key="chat_history",
    #     input_key="question",
    #     output_key="answer"
    #)

uploaded_file = st.file_uploader("上传PDF文件", type="pdf")
question = st.text_input("对PDF的内容进行提问：", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("请输入API密钥！")

if uploaded_file and question and openai_api_key:
    with st.spinner("AI思考中，请稍后..."):
        response = qa_agent(openai_api_key,
                            uploaded_file,
                            question)
        st.write("###答案")
        st.write(response)
        # st.write(response["answer"])
        # st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
