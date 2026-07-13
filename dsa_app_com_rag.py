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

st.set_page_config(page_title="Assistente Jurídico Multiformato", layout="wide")
st.markdown("## ⚖️ Assistente Jurídico Multiformato")

# Sidebar
with st.sidebar:
    api_key = st.text_input("GROQ API Key", type="password")

if not api_key:
    st.warning("Informe a GROQ API Key na lateral.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# Lógica de seleção
opcao = st.radio("Selecione a origem dos dados:", ["Arquivos (PDF, Word, TXT, Excel)", "Website (URL)"])

uploaded_files = None
url_input = None

if opcao == "Arquivos (PDF, Word, TXT, Excel)":
    uploaded_files = st.file_uploader("Upload de documentos", accept_multiple_files=True, type=["pdf", "docx", "txt", "xlsx"])
else:
    url_input = st.text_input("Cole a URL do site:")

# Processamento
def processar_dados(arquivos=None, url=None):
    docs = []
    if url:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    elif arquivos:
        for f in arquivos:
            ext = os.path.splitext(f.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            
            if ext == ".pdf": loader = PyPDFLoader(tmp_path)
            elif ext == ".docx": loader = Docx2txtLoader(tmp_path)
            elif ext == ".txt": loader = TextLoader(tmp_path, encoding='utf-8')
            elif ext == ".xlsx": loader = UnstructuredExcelLoader(tmp_path)
            else: continue
            docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

if (uploaded_files or url_input) and st.button("Processar Dados"):
    with st.spinner("Processando..."):
        st.session_state.db = processar_dados(arquivos=uploaded_files, url=url_input)
        st.success("Base de conhecimento pronta!")

# Chat
if "db" in st.session_state and st.session_state.db:
    pergunta = st.text_area("Sua pergunta:")
    if st.button("Perguntar"):
        retriever = st.session_state.db.as_retriever()
        chain = (
            {"context": retriever | (lambda d: "\n\n".join([doc.page_content for doc in d])), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template("Responda baseando-se no contexto:\n{context}\n\nPergunta: {question}")
            | ChatGroq(model="llama-3.1-70b-versatile")
            | StrOutputParser()
        )
        st.write(chain.invoke(pergunta))
