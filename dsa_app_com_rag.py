import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Importando carregadores estáveis
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader, WebBaseLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Assistente Jurídico", layout="wide")

with st.sidebar:
    api_key = st.text_input("GROQ API Key", type="password")

if not api_key:
    st.warning("Informe a GROQ API Key na barra lateral.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

# Interface de escolha
opcao = st.radio("Origem dos dados:", ["Arquivos", "Website"])
uploaded_files = None
url_input = None

if opcao == "Arquivos":
    uploaded_files = st.file_uploader("Upload de documentos", accept_multiple_files=True, type=["pdf", "docx", "txt", "xlsx"])
else:
    url_input = st.text_input("URL do site")

if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = tempfile.mkdtemp()

def processar_dados(arquivos=None, url=None):
    docs = []
    if url:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    elif arquivos:
        for file in arquivos:
            ext = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            # Carregadores específicos
            if ext == ".pdf": loader = PyPDFLoader(tmp_path)
            elif ext == ".docx": loader = Docx2txtLoader(tmp_path)
            elif ext == ".txt": loader = TextLoader(tmp_path, encoding='utf-8')
            elif ext == ".xlsx": loader = UnstructuredExcelLoader(tmp_path)
            else: continue
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings, persist_directory=st.session_state.chroma_dir)

if (uploaded_files or url_input) and st.button("Processar Dados"):
    with st.spinner("Processando..."):
        st.session_state.vectordb = processar_dados(arquivos=uploaded_files, url=url_input)
        st.session_state.vectordb_ready = True
        st.success("Pronto!")

if st.session_state.get("vectordb_ready"):
    pergunta = st.text_area("Sua pergunta:")
    if st.button("Perguntar"):
        rag_pipeline = RunnableParallel(
            context=(st.session_state.vectordb.as_retriever() | (lambda docs: "\n\n".join([d.page_content for d in docs]))),
            question=RunnablePassthrough()
        ) | ChatPromptTemplate.from_template("Responda com base no contexto: {context}\n\nPergunta: {question}") | llm | StrOutputParser()
        st.write(rag_pipeline.invoke(pergunta))
