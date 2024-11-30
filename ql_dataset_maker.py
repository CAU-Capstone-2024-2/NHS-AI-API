import os
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote
import re

# Define the desired terms to filter files
DESIRED_TERMS = ['급성', '급성고환염', '급성 합병증', '당뇨병케토산증', '저혈당', '당뇨병 합병증(급성 합병증)', '당뇨병 합병증(급성 합병증_저혈당)', '저혈당' '급성부고환염', '급성 간부전', '급성 바이러스 위장관염', '급성신손상(소아)', '급성 세균성 장염', '노로바이러스', '심금연' '심낭염(급성 심낭염)', '당뇨병 합병증(급성 합병증_당뇨병케토뇨증', '고혈당고삼투질상태)', '급성 심근경색증', '급성 충수염', '급성호흡기바이러스감염증', '급성호흡곤란증후군', '심부전', '부정맥', '심장 판막 질환', '대동맥 박리', '심실중격결손증', '동맥관 개존증', '심방중격결손증', '폐색전증', '감염성 심내막염', '심낭염', '고혈압성 심장질환', '협심증', '폐렴', '만성폐쇄성폐질환', '기흉', '부신부전증', '갑상선 기능 항진증', '갑상선 기능 저하증', '갈색세포종', '뇌졸중', '뇌전증', '뇌수막염', '뇌하수체 기능 저하증', '패혈증', '중증열성혈소판감소증후군', '말라리아', '레지오넬라증', '일본뇌염', '광견병', '파상풍', '디프테리아', '백일해', '비브리오 패혈증', '아나필락시스', '독극물 섭취', '영아돌연사증후군', '췌장염', '장결핵', '샤가스병', '바이러스성 출혈열']

def process_xml_file(file_path):
    # Parse XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Get the file number from filename (e.g., "601" from "601_부종.xml")
    file_number = os.path.basename(file_path).split('_')[0]
    
    # Get the condition name from filename (e.g., "부종" from "601_부종.xml")
    condition = os.path.basename(file_path).split('_')[1].replace('.xml', '')
    
    results = []
    processed_names = set()  # Keep track of processed CNTNTS_CL_NM
    
    # Find all cntntsCl elements
    for content in root.findall('.//cntntsCl'):
        # Get CNTNTS_CL_NM
        name_elem = content.find('CNTNTS_CL_NM')
        if name_elem is None or not name_elem.text:
            continue
            
        name = name_elem.text.strip()
        
        # Skip if it contains "참고문헌" or if we've already processed this name
        if "참고문헌" in name or name in processed_names:
            continue
            
        # Add name to processed set
        processed_names.add(name)
        
        # Get CNTNTS_CL_CN
        content_elem = content.find('CNTNTS_CL_CN')
        if content_elem is None or not content_elem.text:
            continue
            
        # Get first sentence from content
        content_text = content_elem.text.strip()
        first_sentence = content_text.split(' ')[0].strip()
        
        # URL encode the name
        # Remove spaces around hyphens
        name_cleaned = re.sub(r'\s*-\s*', '-', name)
        # Replace hyphen with %2D before URL encoding
        name_cleaned = name_cleaned.replace('-', '%2D')
        encoded_name = quote(name_cleaned, safe='%')
        encoded_content = quote(first_sentence, safe='%')
        link = f"https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do?cntnts_sn={file_number}#:~:text={encoded_name},-{encoded_content}"
        
        # Create question
        question = f"{condition} {name}"
        
        # Create result dictionary
        result = {
            "question": question,
            "link": link
        }
        
        results.append(result)
    
    return results

def main():
    # Directory containing XML files
    xml_dir = "./data/documents"
    
    # Output file
    output_file = "./data/QL_dataset.jsonl"
    
    all_results = []
    processed_count = 0
    
    # Process each XML file
    print(f"Looking for files containing these terms: {DESIRED_TERMS}")
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):
            # Check if the filename contains any of the desired terms
            if any(term in filename for term in DESIRED_TERMS):
                file_path = os.path.join(xml_dir, filename)
                print(f"Processing file: {filename}")
                results = process_xml_file(file_path)
                all_results.extend(results)
                processed_count += 1
            
    # Write results to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete:")
    print(f"- Processed {processed_count} files")
    print(f"- Generated {len(all_results)} entries")
    print(f"- Output saved to {output_file}")

if __name__ == "__main__":
    main()