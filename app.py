import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader

# --- Knowledge Base Setup ---
# 1. Load the document
loader = TextLoader("knowledge_base.txt")
docs = loader.load()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 3. Create embeddings for the document chunks
ollama_embeddings = OllamaEmbeddings(model="llama3.2")

# 4. Create a FAISS vector store from the chunks and embeddings
vector_store = FAISS.from_documents(documents, ollama_embeddings)
retriever = vector_store.as_retriever()


# --- LLM and Chain Setup ---
llm = Ollama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
<context>
{context}
</context>
Question: {input}
                                          
Do not use any prior knowledge. If the answer is not contained within the text below, say "I don't know".      
Do not mention anything about the context in your answer. Be concise and clear.                                                                            
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- Streamlit App ---
st.title("Akshay Vedpathak Chatbot")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if user_input := st.chat_input("Ask a question about "):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get response from the retrieval chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = retrieval_chain.invoke({"input": user_input})
            st.write(response["answer"])

    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})



# import streamlit as st
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import AIMessage, HumanMessage

# # Set up the LangChain components
# ollama_llm = ChatOllama(model="llama3.2")
# prompt = ChatPromptTemplate.from_template("Answer the following question concisely:\n{input}")
# chain = prompt | ollama_llm

# # Initialize Streamlit app and session state
# st.title("Akshay Chatbot")

# # Initialize conversation history in Streamlit's session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous messages
# for message in st.session_state.messages:
#     if isinstance(message, HumanMessage):
#         with st.chat_message("user"):
#             st.write(message.content)
#     elif isinstance(message, AIMessage):
#         with st.chat_message("assistant"):
#             st.write(message.content)

# # Get user input
# if user_input := st.chat_input("Ask a question about anything..."):
#     # Add user message to conversation history
#     st.session_state.messages.append(HumanMessage(content=user_input))

#     # Display the user message
#     with st.chat_message("user"):
#         st.write(user_input)

#     # Invoke the chain to get a response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = chain.invoke({"input": user_input})
#             st.write(response.content)
    
#     # Add the AI message to conversation history
#     st.session_state.messages.append(AIMessage(content=response.content))

