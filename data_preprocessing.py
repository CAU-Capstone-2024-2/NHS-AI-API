import os
import json
import xml.etree.ElementTree as ET
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import VoyageEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
from collections import defaultdict

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
        batch_size=128,
        max_retries=30
    ),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=93,
)

def extract_cdata_content(text):
    """Extract content from CDATA sections"""
    if not text:
        return ""
    start = text.find("<![CDATA[")
    end = text.find("]]>")
    if start != -1 and end != -1:
        return text[start + 9:end].strip()
    return text.strip()

def is_url(text):
    """Check if text is a URL"""
    return text.startswith(("http://", "https://"))

def process_xml_content(element, chunks, full_content):
    """Process XML content recursively"""
    # Dictionary to store content by CLSN
    clsn_content = defaultdict(list)
    
    def process_cntntsCl(cl_element):
        """Process a single cntntsCl element"""
        nm = ""
        cn = ""
        clsn = ""
        
        for child in cl_element:
            if child.tag == "CNTNTS_CL_NM":
                nm = extract_cdata_content(child.text)
            elif child.tag == "CNTNTS_CL_CN":
                cn = extract_cdata_content(child.text)
            elif child.tag == "CNTNTSCLSN":
                clsn = extract_cdata_content(child.text)
        
        if nm and cn and clsn and nm != "참고문헌":
            if not is_url(cn) and cn.strip():
                return clsn, (nm, cn)
        return None, None
    
    # First pass: collect all content by CLSN
    for child in element:
        if child.tag == "cntntsCl":
            clsn, content = process_cntntsCl(child)
            if clsn and content:
                clsn_content[clsn].append(content)
        else:
            process_xml_content(child, chunks, full_content)
    
    # Second pass: combine and process content
    for clsn, contents in clsn_content.items():
        if len(contents) > 0:
            nm = contents[0][0]  # Use the first NM
            combined_cn = " ".join(content[1] for content in contents)
            
            content = f"{nm}: {combined_cn}"
            
            # Add to full content
            full_content.append(content)
            
            # Apply semantic chunking if content exceeds 500 characters
            if len(content) > 600:
                sub_chunks = text_splitter.create_documents([content])
                for sub_chunk in sub_chunks:
                    if sub_chunk.page_content.strip():
                        chunks.append(sub_chunk.page_content)
            else:
                chunks.append(content)

def process_document(file_path, doc_number):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    chunks = []
    full_content = []
    process_xml_content(root, chunks, full_content)
    
    doc_id = f"doc_{doc_number}"
    original_uuid = uuid4().hex
    
    processed_chunks = []
    for idx, chunk_content in enumerate(chunks):
        if chunk_content:  # Check if content is not empty
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "original_index": idx,
                "content": chunk_content
            }
            processed_chunks.append(chunk)
    
    result = {
        "doc_id": doc_id,
        "original_uuid": original_uuid,
        "content": "\n\n".join(full_content),
        "chunks": processed_chunks
    }
    
    return result

def main():
    directory = './data/documents'
    output = []
    
    filenames = [f for f in os.listdir(directory) if f.endswith('.xml')]
    filenames.sort()

    for idx, filename in enumerate(filenames, 1):
        file_path = os.path.join(directory, filename)
        processed = process_document(file_path, idx)
        output.append(processed)    
        
    with open('data/doc.json', 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()