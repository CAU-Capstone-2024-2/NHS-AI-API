import json
from openai import OpenAI
from tqdm import tqdm

def load_test_data(test_file):
    """Load test dataset from jsonl file"""
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data

def predict_value(response):
    """Convert response to standardized True/False value"""
    return "True" if "true" in response.lower() else "False"

def run_inference(client, example):
    """Run inference for a single example"""
    text = example["input"].replace("다음 건강 정보 관련 질문이 [급성고환염, 당뇨병 합병증(급성 합병증), 당뇨병 합병증(급성 합병증_저혈당), 급성부고환염, 급성 간부전, 급성 바이러스 위장관염, 급성신손상(소아), 급성 세균성 장염, 심낭염(급성 심낭염), 당뇨병 합병증(급성 합병증_당뇨병케토산증, 고혈당고삼투질상태), 급성 심근경색증, 급성 충수염, 급성호흡기바이러스감염증, 급성호흡곤란증후군, 심부전, 부정맥, 심장 판막 질환, 대동맥 박리, 심실중격결손증, 동맥관 개존증, 심방중격결손증, 폐색전증, 감염성 심내막염, 심낭염, 고혈압성 심장질환, 협심증, 폐렴, 만성폐쇄성폐질환, 기흉, 부신부전증, 갑상선 기능 항진증, 갑상선 기능 저하증, 갈색세포종, 뇌졸중, 뇌전증, 뇌수막염, 뇌하수체 기능 저하증, 패혈증, 중증열성혈소판감소증후군, 말라리아, 레지오넬라증, 일본뇌염, 광견병, 파상풍, 디프테리아, 백일해, 비브리오 패혈증, 아나필락시스, 독극물 섭취, 영아돌연사증후군, 췌장염, 장결핵, 샤가스병, 바이러스성 출혈열] 카테고리 안에 속한다며 True 속하지 않는다면 False을 출력하세요. 다른 내용 없이 True 또는 False만을 출력하세요.: ","")
    a = "급성, 급성고환염, 급성 합병증, 당뇨병케토산증, 저혈당, 당뇨병 합병증(급성 합병증), 당뇨병 합병증(급성 합병증_저혈당), 급성부고환염, 급성 간부전, 급성 바이러스 위장관염, 급성신손상(소아), 급성 세균성 장염, 심낭염(급성 심낭염), 당뇨병 합병증(급성 합병증_당뇨병케토뇨증, 고혈당고삼투질상태), 급성 심근경색증, 급성 충수염, 급성호흡기바이러스감염증, 급성호흡곤란증후군, 심부전, 부정맥, 심장 판막 질환, 대동맥 박리, 심실중격결손증, 동맥관 개존증, 심방중격결손증, 폐색전증, 감염성 심내막염, 심낭염, 고혈압성 심장질환, 협심증, 폐렴, 만성폐쇄성폐질환, 기흉, 부신부전증, 갑상선 기능 항진증, 갑상선 기능 저하증, 갈색세포종, 뇌졸중, 뇌전증, 뇌수막염, 뇌하수체 기능 저하증, 패혈증, 중증열성혈소판감소증후군, 말라리아, 레지오넬라증, 일본뇌염, 광견병, 파상풍, 디프테리아, 백일해, 비브리오 패혈증, 아나필락시스, 독극물 섭취, 영아돌연사증후군, 췌장염, 장결핵, 샤가스병, 바이러스성 출혈열"

    items = a.split(", ")
    target_keywords = items

    for item in target_keywords:
        if item in text:
            return "True"
    #return "False"


    completion = client.with_options(max_retries=1, timeout=2).chat.completions.create(
        model="mldljyh/nhs_1.5b_1_r16_merged_t2",
        messages=[
            {
                "role": "system",
                "content": example["instruction"]
            },
            {
                "role": "user",
                "content": example["input"]
            }
        ],
        temperature=0,
        top_p=0.1
    )
    response = completion.choices[0].message.content.strip()
    """
    if "true" not in response.lower():
        for item in target_keywords:
            if item in text:
                return "True"
    """

    return predict_value(response)

def calculate_metrics(predictions, ground_truth):
    """Calculate accuracy, precision, recall and F1 metrics"""
    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate true positives, false positives, false negatives
    true_positives = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt and p == "True")
    false_positives = sum(1 for p, gt in zip(predictions, ground_truth) if p == "True" and gt == "False")
    false_negatives = sum(1 for p, gt in zip(predictions, ground_truth) if p == "False" and gt == "True")
    
    # Calculate precision, recall and F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total": total
    }

def main():
    # Initialize OpenAI client
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="1"
    )

    # Load test data
    test_file = "test_dataset.jsonl"
    test_data = load_test_data(test_file)
    
    # Run predictions
    predictions = []
    ground_truth = []
    # Open predict file for writing predictions
    with open("qwen_1.5b_bench.jsonl", "w", encoding="utf-8") as predict_file:
        print("Running predictions...")
        for example in tqdm(test_data):
            try:
                prediction = run_inference(client, example)
                predictions.append(prediction)
                ground_truth.append(example["output"])
                
                # Write prediction to jsonl file
                result = {
                    "input": example["input"],
                    "output": prediction
                }
                predict_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)
    
    # Print results
    print("\nBenchmark Results:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1']:.2%}")
    print(f"Correct: {metrics['correct']}")
    print(f"Total: {metrics['total']}")
    
    # Save detailed results
    results = {
        "metrics": metrics,
        "predictions": [
            {
                "input": example["input"],
                "predicted": pred,
                "actual": gt,
                "correct": pred == gt
            }
            for example, pred, gt in zip(test_data, predictions, ground_truth)
        ]
    }
    
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nDetailed results saved to benchmark_results.json")
    print("Predictions saved")

if __name__ == "__main__":
    main()