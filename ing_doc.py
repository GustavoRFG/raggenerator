# ing_doc.py (versão atualizada com suporte a múltiplos uploads acumulativos)

import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Função principal para processar os PDFs enviados por projeto
def processar_documentos_pdf(pasta_projeto: str, caminho_uploads: str = "./uploads"):
    """
    Extrai textos dos PDFs enviados e os vetoriza usando LangChain + Chroma.
    Cada projeto tem sua própria base vetorial salva em vectorstore/{pasta_projeto}.
    Se já existir uma base, novos documentos são adicionados cumulativamente.
    """
    caminho_pdfs = os.path.join(caminho_uploads)
    caminho_chroma = os.path.join("./vectorstore", pasta_projeto)

    if not os.path.exists(caminho_pdfs):
        raise FileNotFoundError(f"A pasta do projeto {pasta_projeto} não foi encontrada em {caminho_pdfs}.")

    documentos = []
    for nome_arquivo in os.listdir(caminho_pdfs):
        if nome_arquivo.endswith(".pdf"):
            caminho_pdf = os.path.join(caminho_pdfs, nome_arquivo)
            loader = PyPDFLoader(caminho_pdf)
            docs = loader.load()
            print(f"📄 {nome_arquivo}: {len(docs)} páginas lidas")
            documentos.extend(docs)

    if not documentos:
        raise ValueError(f"❌ Nenhum texto foi extraído dos PDFs em {caminho_pdfs}. Verifique se os arquivos não estão em imagem escaneada.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documentos)

    if not chunks:
        raise ValueError("❌ A divisão em chunks resultou em lista vazia. Verifique se o conteúdo dos PDFs foi corretamente lido.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(caminho_chroma):
        db = Chroma(persist_directory=caminho_chroma, embedding_function=embeddings)
        db.add_documents(chunks)
        print(f"➕ {len(chunks)} novos chunks adicionados ao projeto '{pasta_projeto}'.")
    else:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=caminho_chroma
        )
        print(f"✅ Nova base vetorial criada para projeto '{pasta_projeto}' com {len(chunks)} chunks.")

    db.persist()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ Uso correto: python ing_doc.py <caminho_para_pdfs> <nome_projeto>")
        sys.exit(1)

    pasta_pdfs = sys.argv[1]
    nome_projeto = sys.argv[2]

    try:
        processar_documentos_pdf(nome_projeto, caminho_uploads=pasta_pdfs)
    except Exception as e:
        print(f"❌ Erro ao processar documentos: {e}")
        sys.exit(1)
