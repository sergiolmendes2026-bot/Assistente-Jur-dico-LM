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

st.set_page_config(page_title="Assistente Jurídico LM", layout="wide")

# Barra Lateral
with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("GROQ API Key", type="password")
    st.markdown("---")
    st.header("Instruções")
    st.markdown("1. Informe sua chave.\n2. Faça o upload dos arquivos.\n3. Pergunte.")
    st.warning("Aviso: a IA pode cometer erros.")

if not api_key:
    st.warning("Informe a GROQ API Key na barra lateral.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

# Upload de Arquivos
uploaded_files = st.file_uploader("Envie documentos (PDF, Word, TXT, Excel)", accept_multiple_files=True, type=["pdf", "docx", "txt", "xlsx"])

if "db" not in st.session_state: st.session_state.db = None

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
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

if uploaded_files and st.button("Processar Arquivos"):
    with st.spinner("Indexando..."):
        st.session_state.db = processar_arquivos(uploaded_files)
        st.success("Arquivos prontos!")

# Chat
if st.session_state.db:
    pergunta = st.text_area("Escreva sua pergunta jurídica")
    if st.button("Perguntar"):
        if not pergunta.strip():
            st.error("Por favor, digite uma pergunta.")
        else:
            retriever = st.session_state.db.as_retriever()
            # O erro de BadRequest muitas vezes vem de contexto vazio ou prompt mal montado
            template = """Use o contexto fornecido para responder à pergunta. Se não souber, diga que não sabe.
            Contexto: {context}
            Pergunta: {question}"""
            chain = (
                {"context": retriever | (lambda d: "\n\n".join([doc.page_content for doc in d])), "question": RunnablePassthrough()}
                | ChatPromptTemplate.from_template(template)
                | llm
                | StrOutputParser()
            )
            try:
                st.write(chain.invoke(pergunta))
            except Exception as e:
                st.error(f"Erro ao consultar IA: {e}")
