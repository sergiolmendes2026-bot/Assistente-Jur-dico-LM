import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader

st.set_page_config(page_title="Assistente Jurídico", layout="wide")

# Barra Lateral com suporte
with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("GROQ API Key", type="password")
    
    if st.button("Clique Aqui Se Precisar de Suporte"):
        st.write("sergiolmendes2026@gmail.com") # Indentação correta aqui

if not api_key:
    st.warning("Por favor, insira sua chave API na barra lateral.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

# Upload de arquivos
uploaded_files = st.file_uploader("Envie seus documentos", accept_multiple_files=True, type=["pdf", "docx", "txt", "xlsx"])

def processar_arquivos(arquivos):
    docs = []
    for f in arquivos:
        ext = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.read())
            path = tmp.name
        
        if ext == ".pdf": loader = PyPDFLoader(path)
        elif ext == ".docx": loader = Docx2txtLoader(path)
        elif ext == ".txt": loader = TextLoader(path, encoding='utf-8')
        elif ext == ".xlsx": loader = UnstructuredExcelLoader(path)
        else: continue
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

if uploaded_files and st.button("Processar Arquivos"):
    with st.spinner("Indexando..."):
        st.session_state.db = processar_arquivos(uploaded_files)
        st.success("Arquivos prontos para consulta!")

if "db" in st.session_state and st.session_state.db:
    pergunta = st.text_area("Sua pergunta jurídica:")
    if st.button("Perguntar"):
        retriever = st.session_state.db.as_retriever()
        chain = (
            {"context": retriever | (lambda d: "\n\n".join([doc.page_content for doc in d])), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template("Responda com base no contexto: {context}\n\nPergunta: {question}")
            | llm
            | StrOutputParser()
        )
        st.write(chain.invoke(pergunta))
