import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import textwrap
import numpy as np
import os
import docx2txt

os.environ["GOOGLE_API_KEY"] = 'AIzaSyBbepUh8x3CqpkxNFnJ1IX0dFc0UNTwwb'
# Set up the embeddings and LLM models
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Ensure 'pdfs' folder exists
if not os.path.exists('pdfs'):
    os.makedirs('pdfs')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

# Function to extract text from a DOCX
def extract_text_from_docx(docx_path):
    try:
        text = docx2txt.process(docx_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

# Function to chunk text
def chunk_text(text, chunk_size=1000):
    try:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        st.error(f"Error chunking text: {str(e)}")
        return []

# Function to embed the text chunks
def embed_text_chunks(text_chunks):
    try:
        return [embeddings.embed_query(chunk) for chunk in text_chunks]
    except Exception as e:
        st.error(f"Error embedding text chunks: {str(e)}")
        return []

# Function to find the most relevant chunk based on cosine similarity
def find_most_relevant_chunk(question_embedding, text_embeddings):
    try:
        similarities = [np.dot(question_embedding, text_embedding) for text_embedding in text_embeddings]
        most_relevant_index = np.argmax(similarities)
        return most_relevant_index
    except Exception as e:
        st.error(f"Error finding relevant chunk: {str(e)}")
        return None

# Function to answer questions based on document content
def answer_question_with_context(question, text_chunks, text_embeddings):
    try:
        question_embedding = embeddings.embed_query(question)
        most_relevant_index = find_most_relevant_chunk(question_embedding, text_embeddings)
        if most_relevant_index is None:
            return "No relevant context found."

        context = text_chunks[most_relevant_index]
        prompt = f"""You are a document analyzer and you will not talk irrelevant to the given context, 
        only based on this Context: {context}\n\nanswer this Question: {question}"""
        response = llm.invoke(prompt)
        content = response.content
        wrapped_content = textwrap.fill(content, width=120)
        return wrapped_content
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return "Error generating response."

def main():
    #centered button
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)     
    with st.sidebar:
        st.title("File Management")
    
        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
        if uploaded_file:
            file_path = os.path.join("pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{uploaded_file.name}' uploaded successfully.")
            st.session_state.selected_file = file_path
    
        # Select existing PDF/DOCX/TXT from the 'pdfs' folder
        existing_files = os.listdir("pdfs")
        st.write("Existing files:", existing_files)  # Debugging line
        if existing_files:
            selected_file = st.selectbox("Choose an existing file", existing_files)
            if selected_file:
                new_file_path = os.path.join("pdfs", selected_file)
                
                # Only reset the session state if a different file is selected
                if 'selected_file' not in st.session_state or st.session_state.selected_file != new_file_path:
                    st.session_state.selected_file = new_file_path
                    st.session_state.full_text = None
                    st.session_state.text_chunks = None
                    st.session_state.text_embeddings = None
                
                if st.button("Proceed"):
                    # Start text extraction and embedding only when "Proceed" is clicked
                    if 'full_text' not in st.session_state or st.session_state.full_text is None:
                        if new_file_path.endswith('.pdf'):
                            st.session_state.full_text = extract_text_from_pdf(st.session_state.selected_file)
                        elif new_file_path.endswith('.docx'):
                            st.session_state.full_text = extract_text_from_docx(st.session_state.selected_file)
                        elif new_file_path.endswith('.txt'):
                            with open(st.session_state.selected_file, 'r') as file:
                                st.session_state.full_text = file.read()
                    
                    if st.session_state.full_text and ('text_chunks' not in st.session_state or st.session_state.text_chunks is None):
                        st.session_state.text_chunks = chunk_text(st.session_state.full_text)
                    
                    if 'text_embeddings' not in st.session_state or st.session_state.text_embeddings is None:
                        if st.session_state.text_chunks:
                            st.session_state.text_embeddings = embed_text_chunks(st.session_state.text_chunks)
        else:
            st.write("No files found in the folder.")
            selected_file = None
    
        # Option to clean existing files
        if st.button("Clear all existing files"):
            try:
                for file in existing_files:
                    os.remove(os.path.join("pdfs", file))
                st.success("All files cleared")
                st.session_state.selected_file = None
                st.session_state.full_text = None
                st.session_state.text_chunks = None
                st.session_state.text_embeddings = None
            except Exception as e:
                st.error(f"Error clearing files: {str(e)}")
    
    # Main UI for querying and response generation
    if 'text_chunks' in st.session_state and st.session_state.text_chunks and 'text_embeddings' in st.session_state and st.session_state.text_embeddings:
        with st.expander("Show Extracted Text"):
            st.write(st.session_state.full_text)
    
    st.title('Document Q-A')
    question = st.text_area("Ask a question about the content")
    
    if st.button("Get Answer"):
        if question and st.session_state.text_chunks and st.session_state.text_embeddings:
            answer = answer_question_with_context(question, st.session_state.text_chunks, st.session_state.text_embeddings)
            st.write(answer)
        else:
            st.error("Please input a question and ensure the file has been processed correctly.")
            
main()
