from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def qa_agent(openai_api_key, uploaded_file, question):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

    #file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_file_path)
    #docs = loader.load()
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n","。","！","？","，","、",""]
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings_model)
    # db = FAISS.from_documents(texts, embeddings_model)
    # retriever = db.as_retriever()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=True
    )

    try:
        result = qa.invoke({"question":question})
        return result["answer"]
    except Exception as e:
        print(f"调用异常：{str(e)}")
        return "处理问题时发生错误"

    # chat_history = []
    # for msg in memory.chat_memory.messages:
    #     if msg.type == "human":
    #         chat_history.append(("Human", msg.content))
    #     elif msg.type == "ai":
    #         chat_history.append(("AI", msg.content))
    #"chat_history": memory,

    # response = qa.invoke({"question": question})
    # return response
