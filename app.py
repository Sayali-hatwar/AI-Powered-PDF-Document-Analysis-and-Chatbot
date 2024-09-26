from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS  # used in vectorembedding tech
from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain.chains.question_answering import load_qa_chain    # helps in chatting
from langchain.prompts import PromptTemplate


from dotenv import load_dotenv
load_dotenv() # loading environment variables


# connect with google api key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
        
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap = 1000)
    chunk = text_splitter.split_text(text)
    return chunk

def get_vector_score(text_chunk):
    embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_score = FAISS.from_texts(text_chunk, embedding)
    vector_score.save_local('faiss_index')  # save info into locals
    
    
def get_conversional_chain():
    prompt_template = """ Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n"""
    model = ChatGoogleGenerativeAI(model = 'gemini-pro',temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ['context','question'])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
# 'stuff' where all documents or chunks are stuff together.
    return chain

def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    new_db = FAISS.load_local('faiss_index',embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversional_chain()
    response = chain({'input_documents':docs,'question':user_question}, return_only_outputs=True)
    print(response)
    st.write('Reply: ', response['output_text'])
    
def main():
    st.set_page_config('chat PDF')
    st.title('Gemini Pro: AI-Powered Document Analysis and Chatbot')
    st.write('This is a simple AI-powered document analysis and chatbot that uses Google Generative AI to interpret and answer questions.')
    
    user_question = st.text_input('Ask question from PDF')
    
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.write('menu: ')
        pdf_docs = st.file_uploader('Upload your PDF and click on submit and process button',accept_multiple_files=True)
        if st.button('submit & process'):
            with st.spinner('processing.....'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_score(text_chunks)
                st.success('PDF processed successfully!')
    

if __name__ == '__main__':
    main()