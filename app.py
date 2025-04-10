from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import json
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from nltk.corpus import wordnet as wn
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import subprocess

# Carrega .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializa o app
app = Flask(__name__)
CORS(app)

# Pastas padrão
UPLOAD_FOLDER = "uploads"
CHROMA_BASE_PATH = "vectorstore"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_BASE_PATH, exist_ok=True)

# Contador de acessos global
access_count = 0

# Função para registrar acessos com IP e pergunta
def log_access(client_ip, question, project="default"):
    global access_count
    access_count += 1
    log_entry = f"Acesso {access_count} | IP: {client_ip} | Projeto: {project} | Pergunta: {question}\n"
    with open("access_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)
    print(log_entry.strip())

# Upload de PDF + ingestao
@app.route("/upload", methods=["POST"])
def upload():
    if 'project' not in request.form or 'files' not in request.files:
        return jsonify({"error": "Projeto ou arquivos ausentes."}), 400

    project = secure_filename(request.form['project'])
    files = request.files.getlist('files')
    project_dir = os.path.join(UPLOAD_FOLDER, project)
    os.makedirs(project_dir, exist_ok=True)

    for file in files:
        if file.filename.endswith(".pdf"):
            file_path = os.path.join(project_dir, secure_filename(file.filename))
            file.save(file_path)
            print(f"📄 PDF salvo em: {file_path}")

    try:
        subprocess.run(["python", "ing_doc.py", project_dir, project], check=True)
        return jsonify({"status": "PDFs processados com sucesso.", "project": project})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Erro ao processar PDFs: {str(e)}"}), 500

# Inicializa banco vetorial
def get_vectordb(project):
    project_path = os.path.join(CHROMA_BASE_PATH, project)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(persist_directory=project_path, embedding_function=embeddings)
    return vectordb

# Inicializa cadeia RAG
def get_qa_chain(project):
    vectordb = get_vectordb(project)
    llm = ChatOpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        return_source_documents=True
    )

# FAQ tradicional (default)
faqs = {
    "exemplo de pergunta": "Esta é uma resposta de exemplo. Substitua com conteúdo real."
}
faq_keys = list(faqs.keys())
vectorizer = TfidfVectorizer().fit(faq_keys)
faq_vectors = vectorizer.transform(faq_keys)

def expand_question_with_synonyms(question):
    words = question.split()
    expanded_words = []
    for word in words:
        synonyms = set()
        for syn in wn.synsets(word, lang="spa"):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        if synonyms:
            expanded_words.append(word + " " + " ".join(synonyms))
        else:
            expanded_words.append(word)
    return " ".join(expanded_words)

def buscar_em_faq(pergunta: str):
    pergunta_vector = vectorizer.transform([pergunta.lower()])
    similarities = cosine_similarity(pergunta_vector, faq_vectors).flatten()
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    if best_score > 0.4:
        return faqs[faq_keys[best_index]]
    return None

@app.route("/rag", methods=["POST"])
def handle_question():
    data = request.get_json()
    question = data.get("question", "").strip()
    project = data.get("project", "default")

    if not question:
        return jsonify({"error": "Pergunta vazia."}), 400

    client_ip = request.remote_addr
    log_access(client_ip, question, project)
    print(f"Pergunta recebida: {question}")
    print(f"Número de acessos: {access_count}")

    try:
        qa_chain = get_qa_chain(project)
        result = qa_chain({"query": question})
        resposta = result["result"]
        sources = result["source_documents"]
        trechos = [doc.page_content for doc in sources]

        print(f"Resposta: {resposta}")
        # print(f"Chunks utilizados: {trechos}")

        if not resposta or resposta.lower().startswith("desculpe") or resposta.lower().startswith("não encontrei"):
            resposta_faq = buscar_em_faq(question)
            if resposta_faq:
                return jsonify({"response": resposta_faq, "fonte": "FAQ"})

        return jsonify({
            "response": resposta,
            "fonte": "RAG",
            "chunks_utilizados": trechos
        })
    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"error": "Erro ao processar a pergunta."}), 500

if __name__ == '__main__':
    print("\n📡 Iniciando o servidor RAG Universal na porta 5001...")
    app.run(debug=True, port=5001, host='0.0.0.0')
