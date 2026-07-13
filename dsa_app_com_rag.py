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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader, WebBaseLoader

# Configuração da página
st.set_page_config(page_title="Assistente Jurídico LM", layout="wide")

# Barra Lateral (Layout conforme solicitado)
with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("Coloque aqui sua GROQ API Key e pressione Enter", type="password")
    
    st.markdown("---")
    st.header("Instruções")
    st.markdown("1. Informe sua chave no campo acima.")
    st.markdown("2. Digite sua pergunta ou dúvida.")
    st.markdown("3. Clique em Enviar.")
    
    st.warning("Aviso: a IA pode cometer erros. Verifique fatos críticos.")
    
    if st.button("📧 Clique Aqui Se Precisar de Suporte"):
        st.write("Suporte: [sergiolmendes2026@gmail.com]")

# Validação da API
if not api_key:
    st.markdown("# ⚖️ Assistente Jurídico LM")
    st.warning("Informe a GROQ API Key na barra lateral para continuar.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# Conteúdo Principal
st.markdown("# ⚖️ Assistente Jurídico")
opcao = st.radio("Selecione a origem dos dados:", ["Arquivos (PDF, Word, TXT, Excel)", "Website (URL)"])

uploaded_files = None
url_input = None

if opcao == "Arquivos (PDF, Word, TXT, Excel)":
    uploaded_files = st.file_uploader("Envie seus documentos", accept_multiple_files=True, type=["pdf", "docx", "txt", "xlsx"])
else:
    url_input = st.text_input("Cole a URL do site:")

# Inicialização e Processamento
if "db" not in st.session_state: st.session_state.db = None

def processar_dados(arquivos, url):
    docs = []
    if url:
        docs.extend(WebBaseLoader(url).load())
    elif arquivos:
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
    return docs

if (uploaded_files or url_input) and st.button("Processar Dados"):
    with st.spinner("Indexando..."):
        docs = processar_dados(uploaded_files, url_input)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.db = Chroma.from_documents(chunks, embeddings)
        st.success("Documentos carregados!")

# Chat
if st.session_state.db:
    pergunta = st.text_area("Escreva sua pergunta jurídica", placeholder="Ex.: Quais cláusulas tratam de rescisão e multas?")
    if st.button("Perguntar"):
        chain = (
            {"context": st.session_state.db.as_retriever() | (lambda d: "\n\n".join([doc.page_content for doc in d])), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template("Responda com base no contexto: {context}\n\nPergunta: {question}")
            | ChatGroq(model="llama-3.1-70b-versatile")
            | StrOutputParser()
        )
        st.write(chain.invoke(pergunta))
