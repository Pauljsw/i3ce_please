import argparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_CHAR_LENGTH = 1000

def extract_query_from_prompt(full_prompt: str):
    start_tag = "[INSPECTION TASKS]"
    if start_tag in full_prompt:
        return full_prompt.split(start_tag, 1)[1].strip()
    return full_prompt.strip()

def generate_rag_prompt(full_prompt: str, output_path: str = "rag_output.txt"):  # ✅ 수정
    embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "/home/aimgroup/ChoSW/Shapellm/SW/ShapeLLM/llava/serve/vectorstore/faiss_index_bridges",
        embeddings=embedding
    )

    query = extract_query_from_prompt(full_prompt)  # ✅ 수정

    docs = vectorstore.similarity_search(query, k=5)

    rag_chunks = []
    total_length = 0
    for doc in docs:
        chunk = doc.page_content
        if total_length + len(chunk) > MAX_CHAR_LENGTH:
            break
        rag_chunks.append(chunk)
        total_length += len(chunk)

    rag_context = "\n".join(rag_chunks)
    print(f"[INFO] 📏 누적 문서 길이: {total_length}자")

    final_prompt = f"[CONTEXT]\n{rag_context}\n\n[QUESTION]\n{query}"

    output_path = os.path.join(SCRIPT_DIR, output_path)
    with open(output_path, "w") as f:
        f.write(final_prompt)

    print(f"[✅] RAG 기반 prompt 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--output", type=str, default="rag_output.txt")
    args = parser.parse_args()

    generate_rag_prompt(args.query, args.output)  # ✅ 이제 query는 full prompt 전체
