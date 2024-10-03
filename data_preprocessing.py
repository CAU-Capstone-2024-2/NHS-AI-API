import os
import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import VoyageEmbeddings
from uuid import uuid4
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 Voyage API 키 가져오기
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY 환경 변수가 .env 파일에 설정되지 않았습니다.")

# SemanticChunker 초기화
text_splitter = SemanticChunker(
    VoyageEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model="voyage-3",
        show_progress_bar=True,
        max_retries=30
    ), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=87,
)

def process_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    documents = text_splitter.create_documents([content])
    
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    original_uuid = uuid4().hex
    chunks = []
    
    for idx, doc in enumerate(documents):
        if doc.page_content:  # Check if content is not empty
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "original_index": idx,
                "content": doc.page_content
            }
            chunks.append(chunk)
    
    result = {
        "doc_id": doc_id,
        "original_uuid": original_uuid,
        "content": content,
        "chunks": chunks
    }
    
    return result

def main():
    # doc_{n}.txt 파일이 있는 디렉토리 설정
    directory = './data/documents'
    output = []
    
    for filename in os.listdir(directory):
        if filename.startswith('doc_') and filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            processed = process_document(file_path)
            output.append(processed)
    
    # 결과를 JSON 파일로 저장
    with open('output.json', 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
