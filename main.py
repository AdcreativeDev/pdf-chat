# Import necessary modules
import re
import time
from io import BytesIO
from typing import Any, Dict, List
import os
import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import datetime
from PIL import Image

st.set_page_config(
    page_title="PDF chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

image = Image.open("ironman-banner.jpg")
st.image(image, caption='created by MJ')

st.title("🤖 :blue[PDF chatbot]")


with st.sidebar:    
    system_openai_api_key = os.environ.get('OPENAI_API_KEY')
    system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
    os.environ["OPENAI_API_KEY"] = system_openai_api_key

    st.markdown(
        """
        ## Features
        1. read local pdf to Vector Store (FAISS)
        2. split documents into pages
        3. using RetrievalQA Chain

        """ )
    
    



# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# Define a function for the embeddings
@st.cache_data
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=system_openai_api_key)
    # Indexing
    # Save in a Vector DB
    with st.spinner("Creating FASS Vectors ..."):
        index = FAISS.from_documents(pages, embeddings)
    now = datetime.datetime.now()
    st.caption(f'✔️ {now.strftime("%H-%M-%S")} : Embedding Created')

    return index





# Allow the user to upload a PDF file
uploaded_file = st.file_uploader(" **Step 1 : Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    if pages:
        # Allow the user to select a page and view its content
        with st.expander("Want to see the PDF Content", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]
        if system_openai_api_key:
            # Test the embeddings and save the index in a vector database
            index = test_embed()
            # Set up the question-answering system
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=system_openai_api_key),
                chain_type = "map_reduce",
                retriever=index.as_retriever(),
            )
            # Set up the conversational agent
            tools = [
                Tool(
                    name="Smart AI robot",
                    func=qa.run,
                    description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                )
            ]
            prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available.
                        If you dont know the query or question, just reply I am stupid and I dont know ! Do not tell the lie.
                        You have access to a single tool:"""
            suffix = """Begin!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history"
                )

            llm_chain = LLMChain(
                llm=OpenAI(
                    temperature=0, openai_api_key=system_openai_api_key, model_name="gpt-3.5-turbo"
                ),
                prompt=prompt,
            )
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
            )

            # Allow the user to enter a query and generate a response
            query = st.text_input(
                "**What's on your mind?**",
                placeholder="Ask me anything from {}".format(name_of_file),
            )

            if query:
                with st.spinner(
                    "Generating Answer to your Query : `{}` ".format(query)
                ):
                    res = agent_chain.run(query)
                    st.info(res, icon="🤖")

            # Allow the user to view the conversation history and other information stored in the agent's memory
            with st.expander("Message History"):
                st.session_state.memory

