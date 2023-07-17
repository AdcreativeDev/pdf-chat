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
    page_title="Ask your PDF",
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

st.title("😀 :blue[Ask your PDF]")


# with st.sidebar:    

#     st.markdown(
#         """
#         ## Features
#         1. read local pdf to Vector Store (FAISS)
#         2. split documents into pages
#         3. using RetrievalQA Chain

#         """ )
    
    



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
def create_embed_vectors():

    embeddings = OpenAIEmbeddings(openai_api_key=system_openai_api_key)
    st.write('✔️ OpenAI Embeddings Created')
    # Indexing
    # Save in a Vector DB
    now = datetime.datetime.now()
    st.write(f'✔️ {now.strftime("%H-%M-%S")} : Start to create FAISS VectorStore for {str(len(pages))} page(s)')
    with st.spinner("Creating FASS Vectors ..."):
        index = FAISS.from_documents(pages, embeddings)
    now = datetime.datetime.now()
    st.write(f'✔️ {now.strftime("%H-%M-%S")} : VectorStore Created')

    return index



system_openai_api_key = os.environ.get('OPENAI_API_KEY')
system_openai_api_key = st.text_input(":key: OpenAI Key :", value=system_openai_api_key)
os.environ["OPENAI_API_KEY"] = system_openai_api_key

with st.expander("Sample"):
    image = Image.open("PdfChat-sample-01.jpg")
    st.image(image, caption='sample')
    image = Image.open("PdfChat-sample-02.jpg")
    st.image(image, caption='sample')
    image = Image.open("PdfChat-sample-03.jpg")
    st.image(image, caption='sample')

st.subheader('Step 1 : Upload Your PDF File')
# Allow the user to upload a PDF file
uploaded_file = st.file_uploader(" **Select PDF File**", type=["pdf"])
try:
    if uploaded_file:
        name_of_file = uploaded_file.name

        doc = parse_pdf(uploaded_file)
        st.write(f'✔️ Parsed file : {name_of_file} Completed')
        pages = text_to_docs(doc)
        st.write(f'✔️ {str(len(pages))} page(s) created after Chunking')

        if pages:
            # Allow the user to select a page and view its content
            with st.expander("Extracted Pages", expanded=False):

                page_sel = st.number_input(
                    label="Select Page", min_value=1, max_value=len(pages), step=1
                )
                pages[page_sel - 1]


            if system_openai_api_key:
                # Test the embeddings and save the index in a vector database
                index = create_embed_vectors()
                # Set up the question-answering system

                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=system_openai_api_key),
                    chain_type = "map_reduce",
                    retriever=index.as_retriever(),
                )
                st.write('✔️ LangChain QA Retreiver Created')


                # Set up the conversational agent
                tools = [
                    Tool(
                        name="Smart AI robot",
                        func=qa.run,
                        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                    )
                ]
                prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available.
                            Case 1 - You can find the answer, reply in detail by point format.

                            Case 2 - If you dont know the query or question or you cannot find the answer. 
                            you must reply this statement : 🥹 Sorry, I do not understand your question or it is not  related to this document, Please kindly re-enter question 🤗.
                            
                            Case 3 - If you cannot find the answer.
                            You must reply this statemtent :🥹 Sorry, I can not find the anwwer from the content 🙄 ! 

                            ATTENTION : Do not generate unrelated result.
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
                    llm=OpenAI( temperature=0, openai_api_key=system_openai_api_key, model_name="gpt-3.5-turbo"),
                    prompt=prompt,
                )
                agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
                agent_chain = AgentExecutor.from_agent_and_tools(
                    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
                )
                st.write('✔️ Chain Agent created ')

                st.write('✔️ Await for your question ')
                st.subheader('Step 2 : Enter your question:')
                # Allow the user to enter a query and generate a response
                query = st.text_input(
                    "Question : ",
                    placeholder="the question should be clear in structure english, and related to the content {}".format(name_of_file),
                )

                if query:
                    with st.spinner(
                        "Generating Answer to your Query : `{}` ".format(query)
                    ):
                        now = datetime.datetime.now()
                        st.write(f'✔️ {now.strftime("%H-%M-%S")} : Query Start)')

                        result  = agent_chain.run(query)
                        
                        now = datetime.datetime.now()
                        st.write(f'✔️ {now.strftime("%H-%M-%S")} : Query Completed)')

                        st.subheader('Result:')
                        st.info(result)

                # Allow the user to view the conversation history and other information stored in the agent's memory
                with st.expander("Message History"):
                    st.session_state.memory
except Exception as e:
    st.error(e, icon="❌")


log = """

Q: tell me if I am qualify for the job if I only have 1.5 working experience ?
> Entering new  chain...
Thought: I need to determine if having 1.5 years of working experience qualifies someone for a job.
Action: Smart AI robot
Action Input: "Does having 1.5 years of working experience qualify someone for a job?"
Observation:  No, having 1.5 years of working experience does not qualify someone for the job.
Thought:I now know the final answer
Final Answer: No, having 1.5 years of working experience does not qualify someone for the job.



Q: List out the requirements (experience, education, language, travel ,age, communication skill) in numeric form, and is this job welcome fresh graduated students ?
> Entering new  chain...
Thought: I need to list out all the requirements in numeric form and determine if fresh graduates are welcome for this job.
Action: Smart AI robot
Action Input: "List out all requirements for an event management/marketing position (experience, education, language, travel, age, communication skill) in numeric form and is this job welcome fresh graduated students?"
Observation:  Fresh graduates are not specified as welcome for this job.
Thought:I have listed out all the requirements in numeric form and determined that fresh graduates are not specified as welcome for this job.
Final Answer: The requirements for an event management/marketing position in numeric form are: 
1. University degree in event management/marketing or a related discipline
2. At least three years of solid experience in event management of large scale outdoor events
3. Experience of liaison with government departments for venue operation and license application
4. Initiative and self-discipline
5. Good communication and organizational skills
6. Proficiency in PC applications
7. Ability to deliver quality work under pressure
8. Ability to work on weekends
9. Solely responsible for end to end event management
10. Maintaining up-to-date databases and comprehensive records
11. Two to three years of relevant working experience
12. Knowledgeable in project management and budgeting
13. Ability to work under pressure

Fresh graduates are not specified as welcome for this job.

> Finished chain.




Q: Who is the employer ? Any contact information such as : email, phone no or website ?  Does it mention about the benefits? 
> Entering new  chain...
Thought: I need to find the employer, contact information, and information about benefits.
Action: Smart AI robot
Action Input: Search for the employer, contact information, and benefits for the event management/marketing position.
Observation:  The employer is Hong Kong Convention and Exhibition Centre (Management) Limited, their contact information can be found on their website http://www.discoverhongkong.com, and the benefits for the event management/marketing position include dental insurance, gratuity, medical insurance, five-day work week, education allowance, flexible working hours.
Thought:I have found the employer, contact information, and information about benefits.
Final Answer: The employer for the event management/marketing position is Hong Kong Convention and Exhibition Centre (Management) Limited. Their contact information can be found on their website http://www.discoverhongkong.com. The benefits for the position include dental insurance, gratuity, medical insurance, five-day work week, education allowance, and flexible working hours.

> Finished chain.



"""
with st.expander("Log"):
    st.code(log)



