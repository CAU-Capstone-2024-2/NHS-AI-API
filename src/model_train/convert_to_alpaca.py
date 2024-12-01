import json
import random
from collections import Counter

def get_label_distribution(data):
    labels = [entry["output"] for entry in data]
    return Counter(labels)

def convert_to_alpaca_format(input_file, train_file, test_file, test_ratio=0.1):
    alpaca_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            alpaca_entry = {
                "instruction": "",
                "input": f"다음 건강 정보 관련 질문이 [급성고환염, 당뇨병 합병증(급성 합병증), 당뇨병 합병증(급성 합병증_저혈당), 급성부고환염, 급성 간부전, 급성 바이러스 위장관염, 급성신손상(소아), 급성 세균성 장염, 심낭염(급성 심낭염), 당뇨병 합병증(급성 합병증_당뇨병케토산증, 고혈당고삼투질상태), 급성 심근경색증, 급성 충수염, 급성호흡기바이러스감염증, 급성호흡곤란증후군, 심부전, 부정맥, 심장 판막 질환, 대동맥 박리, 심실중격결손증, 동맥관 개존증, 심방중격결손증, 폐색전증, 감염성 심내막염, 심낭염, 고혈압성 심장질환, 협심증, 폐렴, 만성폐쇄성폐질환, 기흉, 부신부전증, 갑상선 기능 항진증, 갑상선 기능 저하증, 갈색세포종, 뇌졸중, 뇌전증, 뇌수막염, 뇌하수체 기능 저하증, 패혈증, 중증열성혈소판감소증후군, 말라리아, 레지오넬라증, 일본뇌염, 광견병, 파상풍, 디프테리아, 백일해, 비브리오 패혈증, 아나필락시스, 독극물 섭취, 영아돌연사증후군, 췌장염, 장결핵, 샤가스병, 바이러스성 출혈열] 카테고리 안에 속한다며 True 속하지 않는다면 False을 출력하세요. 다른 내용 없이 True 또는 False만을 출력하세요.: {data['question']}",
                "output": "True" if data['label'] else "False"
            }
            
            alpaca_data.append(alpaca_entry)
    
    # Shuffle and split data while maintaining distribution
    random.shuffle(alpaca_data)
    
    # Calculate split index
    total_size = len(alpaca_data)
    test_size = int(total_size * test_ratio)
    
    # Split data
    test_data = alpaca_data[:test_size]
    train_data = alpaca_data[test_size:]
    
    # Verify distribution
    orig_dist = get_label_distribution(alpaca_data)
    train_dist = get_label_distribution(train_data)
    test_dist = get_label_distribution(test_data)
    
    print(f"Original distribution: {dict(orig_dist)}")
    print(f"Train distribution: {dict(train_dist)}")
    print(f"Test distribution: {dict(test_dist)}")
    
    # Write to output files
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = "after_filtered_classify_dataset.jsonl"
    train_file = "train_dataset.jsonl"
    test_file = "test_dataset.jsonl"
    convert_to_alpaca_format(input_file, train_file, test_file)
    print(f"Conversion completed. Train data saved to {train_file}, Test data saved to {test_file}")