import openai
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI


openai.api_key = os.getenv('openai_api_key')

def process_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_summary(uploaded_file):
    # Load Document if file is uploaded
    if uploaded_file is not None:
        documents = process_pdf(uploaded_file)
        #Split Documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=150)
        document_content = text_splitter.split_text(documents)
        docs = text_splitter.create_documents(document_content)
        
        #Define Prompt
        prompt_template = """Write a 200 word summary """
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Define LLM Chain
        llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key, model_name="gpt-3.5-turbo")
        summary_chain = load_summarize_chain(llm, chain_type="refine")
        
        # summary_list = []

        # for i, doc in enumerate(docs):
            
        #     chunk_summary = summary_chain.run([doc])
        #     summary_list.append(chunk_summary)
        
        chunk_summary = summary_chain.run(docs)
        return chunk_summary
    
# Page title
st.title('PDF Summarizer')


with st.form('myform', clear_on_submit=True):
    # File Upload
    uploaded_file = st.file_uploader('Upload a Pdf File', type='pdf')
    submitted = st.form_submit_button('Summarize')
    if submitted:
        with st.spinner('Summarizing...'):
            summarys = generate_summary(uploaded_file)
            st.write("Summaries:")
            # for summary in summarys:
            st.write(summarys)
            
