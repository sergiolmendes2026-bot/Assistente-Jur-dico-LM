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

# Carregadores específicos e estáveis
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader, WebBaseLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Assistente Jurídico LM", layout="wide")
st.markdown("## ⚖️ Assistente Jurídico LM (Multiformato)")

with st.sidebar:
    api_key = st.text_input("GROQ API Key", type="password")

if not api_key:
    st.warning("Informe a GROQ API Key na barra lateral.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

# Interface de Input
col_tipo = st.radio("Selecione a origem dos dados:", ["Arquivos (PDF, Word, TXT, Excel)", "Website (URL)"])

uploaded_files = None
url_input = None

if col_tipo == "Arquivos (PDF, Word, TXT, Excel)":
    uploaded_files = st.file_uploader("Envie seus documentos", type=["pdf", "docx", "txt", "xlsx"], accept_multiple_files=True)
else:
    url_input = st.text_input("Cole a URL do site (ex: https://site.com)")

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
            
            if ext == ".pdf": loader = PyPDFLoader(tmp_path)
            elif ext == ".docx": loader = Docx2txtLoader(tmp_path)
            elif ext == ".txt": loader = TextLoader(tmp_path, encoding='utf-8')
            elif ext in [".xlsx", ".xls"]: loader = UnstructuredExcelLoader(tmp_path)
            else: continue
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    # Modelo de embedding leve e estável
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings, persist_directory=st.session_state.chroma_dir)

if (uploaded_files or url_input) and st.button("Processar Dados"):
    with st.spinner("Processando..."):
        st.session_state.vectordb = processar_dados(arquivos=uploaded_files, url=url_input)
        st.session_state.vectordb_ready = True
        st.success("Pronto para consulta!")

if st.session_state.get("vectordb_ready"):
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
    pergunta = st.text_area("Sua pergunta")
    if st.button("Perguntar"):
        rag_pipeline = RunnableParallel(
            context=(retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))),
            question=RunnablePassthrough()
        ) | ChatPromptTemplate.from_template("Responda com base no contexto: {context}\n\nPergunta: {question}") | llm | StrOutputParser()
        st.markdown("### Resposta")
        st.write(rag_pipeline.invoke(pergunta))
