import json
import os
import queue
import threading
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Medical conditions list
target_conditions = ["사람의 심장은 크게 왼쪽과 오른쪽으로 나뉘며, 각 부분은 다시 심방과 심실로 나뉩니다.", "갈색세포종은 부신수질에서 카테콜아민을 과다 분비하는 종양으로, 모든 갈색세포종은 잠재적으로 전이의 위험이 있어 악성 종양으로 분류됩니다.", "고환은 음낭 안에 위치하는 내분비기관으로 한 쌍의 달걀 모양 구조를 가지고 있으며, 정자와 남성호르몬(테스토스테론 )을 생산합니다.", "부고환은 고환의 후외측에 위치하는 5 cm 길이의 기관으로 고환에서 만들어진 정자가 이곳을 통과하면서 운동능력을 갖게 되어 비로소 수태 능력을 갖추게 됩니다.", "당뇨병의 급성 합병증은 체내 혈당이 너무 높거나 낮을 때 생길 수 있습니다.", "당뇨병케토산증(diabetic ketoacidosis)과 고혈당고삼투질상태(hyperglycemic hyperosmolar state)는 일반적으로 심한 고혈당과 인슐린 결핍이 있을 때 발생합니다.", "저혈당은 약물치료를 받는 당뇨병 환자가 가장 흔히 겪는 합병증입니다.", "심낭은 심장을 싸고 있는 두 겹의 막(벽심장막, 내장심장막)으로 이루어진 주머니입니다.", "급성 심낭염은 여러 가지 질병과 다양한 요인에 의해 심장막에 염증이 생긴 상태입니다.", "만성 심낭삼출은 급성 심낭염의 병력이 없는 사람에서도 적지 않게 볼 수 있습니다.", "심장눌림증은 심장을 둘러싼 심낭 안에 여러가지 원인에 의해서 축적된 체액이나 혈액의 양이 증가해 심장을 압박하고, 그로 인해 심장 내로 전신 순환에 필요한 혈액이 충분히 들어오지 못하는 상태입니다.", "교착 심낭염은 급성 심낭염이나 만성 심낭삼출이 치유되는 과정에서 심장막이 서서히 섬유화되면서 두꺼워지고, 때로는 칼슘 성분이 침착되어 심장막이 딱딱해지고 뻣뻣해지는 병입니다.", "심장은 우리 몸의 세포 구석구석까지 피를 공급하는 장기이며 정상 성인의 경우 대략 분당 5~6리터의 피를 뿜어냅니다.", "심장은 혈액 순환을 유지하는 장기로, 심실이 이완되면서 전신과 폐에서 혈액을 받아들이는 충만 작용과 심실이 수축하면서 전신 및 폐로 혈액을 내보내는 박출 작용을 합니다.", "만성폐쇄성폐질환(병태생리학적·임상적 의미로서 정의)은 만성기관지염이나 폐기종에 의하여 기류폐색 소견이 관찰되는 질환군으로, 기류폐색은 대부분 비가역적이고 진행성이나, 일부에서는 기도 과민성이 동반되거나 기도폐색이 부분적으로 가역성일 수도 있습니다.", "대표적인 인수공통감염병입니다.", "급성호흡기감염증은 주로 바이러스나 세균에 의해 발생하지만, 면역이 심하게 떨어지면 드물게 진균(곰팡이)에 의해 발생할 수도 있습니다.", "중증열성혈소판감소증후군(severe fever with thrombocytopenia syndrome, SFTS)이란 중증열성혈소판증후군 바이러스에 의한 감염병입니다.", "패혈증(敗血症, sepsis)은 미생물 감염이 원인이 되어, 인체의 감염되지 않은 다른 부위에까지 심각한 영향이 생기는 상태입니다.", "심장은 온몸에 혈액을 보내 산소와 영양소를 전달하는 펌프로, 평생 단 한 순간도 쉬지 않고 일합니다.", "말라리아는 플라스모디움 속(genus Plasmodium)에 속하는 기생충이 척추동물의 적혈구에 기생하여 발생하는 감염 질환입니다.", "심장에는 여닫이문 역할을 하는 4개의 판막이 있습니다.", "췌장은 이자라고도 불리며 소화를 담당하는 장기입니다.", "폐렴은 호흡기(폐)가 병원체에 감염되어 염증이 발생하는 질환입니다.", "부신은 양쪽 콩팥 위에 위치하는 대표적인 내분비샘입니다.", "세균 감염에 의한 급성위장관염을 일컫습니다.", "레지오넬라증은 레지오넬라속(屬, genus) 세균에 의한 감염증으로 레지오넬라 폐렴(Legionnaire's disease, 재향군인병, 레지오넬라병)과 특별한 치료 없이도 호전되는 급성 열성 질환인 폰티악 열(pontiac fever) 등 두 가지 질병을 합쳐서 일컫는 말입니다.", "사람의 중추신경계는 뇌와 척수로 구성되며 '뇌척수막' 이라는 세 겹의 막에 싸여 보호받고 있습니다.", "백일해균(Bordetella pertussis) 감염에 의한 급성 호흡기질환으로, 호흡기 분비물이나 비말을 통한 호흡기 전파가 주된 감염경로입니다.", "급성 감염성 위장관염은 모든 연령에서 발생하는 매우 흔한 질환입니다.", "파상풍은 상처 부위에서 파상풍균(Clostridium tetani)이 생산한 신경 독소에 의해 근육수축과 통증이 나타나는 감염성 질환입니다.", "디프테리아는 디프테리아균(Corynebacterium diphtheriae) 감염 후 발생하는 급성, 독소(toxin) 매개성 호흡기 감염병으로 인체의 모든 점막을 침범할 수 있으며 침범부위에 막(membrane)이 생기는 것이 특징입니다.", "충수염은 진료 현장에서 수술이 필요한 복통의 가장 흔한 원인이며, 매년 우리나라에서 10만명 이상이 급성 충수염으로 수술을 받습니다.", "결핵균(Mycobacterium tuberculosis)은 폐뿐 아니라 위장관, 관절, 뇌수막, 심낭, 비뇨생식기계 등 많은 장기를 침범할 수 있는데 이를 폐외결핵이라 하며 전체 결핵의 약 10-15%를 차지합니다.", "비브리오 패혈증은 비브리오 불니피쿠스(Vibrio vulnificus) 세균 감염에 의하여 급성 패혈증이 발생하는 질병입니다.", "심장은 온몸에 혈액을 순환시키는 펌프 역할을 하면서 산소와 영양소를 전달합니다.", "영아 돌연사 증후군(Sudden infant death syndrome)은 1세 이전의 영아가 갑자기 예상치 못하게 사망했으나 완전한 부검과 현장 조사, 과거의 병이 있었는지를 조사했는데도 원인을 알 수 없는 경우를 말합니다.", "폐렴은 말단세기관지 아래 폐실질에 염증이 생기는 병입니다.", "폐는 호흡을 통해 혈액에 산소를 공급해주고 혈액 속에서 이산화탄소를 배출하는 역할을 하는 장기입니다.", "뇌졸중은 뇌혈관 이상에 의해 발생하는 질환입니다.", "샤가스병은 크루스파동편모충(Trypanosoma cruzi) 감염에 의한 질환입니다.", "동맥관이란 대동맥과 폐동맥 사이를 연결하는 혈관으로 출생 전에 존재했다가 출생과 함께 닫힙니다.", "심방중격결손은 우심방과 좌심방 사이의 벽(심방중격)에 구멍이 있는 선천성 심장병입니다.", "간은 우리 몸에서 가장 큰 기관으로 무게가 약 1', '2~1', '5 kg이고 우상복부(배의 오른쪽 윗부분)에 위치하며 갈비뼈에 의해 보호되고 있습니다", "사람이 삶을 유지하기 위해서 여러 장기가 정상적으로 기능해야 하지만, 특히 호흡을 담당하는 폐는 생존을 위해 필수적인 장기로 흔히 사람이 운명했을 때, '숨이 끊어졌다' '숨졌다'고 표현합니다.", "호흡기관인 폐는 우리가 사는데 꼭 필요한 산소를 흡수하고 몸에서 발생한 노폐물인 이산화탄소를 배출하는 역할을 합니다.", "대동맥은 내막, 중막, 외막 3개의 층으로 구성되어 있습니다.", "우리 뇌의 신경세포는 정상적으로 늘 전기를 띠고 있습니다.", "알레르기란 주위 환경에 존재하는 항원(원인 물질)에 대한 면역 매개형 과민반응입니다.", "모기를 통해 전파되는 감염병중 하나로 일본뇌염 바이러스(Japanese encephalitis virus)가 원인입니다.", "우리는 일상생활에서 많은 화학물질과 의약품을 사용합니다.", "뇌졸중이란 뇌혈류의 장애로 인해 발생하는 갑작스런 뇌세포 손상으로, 신경학적 증상이 발생하여 24시간 이상 지속되는 경우를 말합니다.", "급성신손상이란 콩팥(신장)콩팥 기능이 갑자기 떨어지는 상태입니다."]



def write_to_jsonl(filename, item):
    """Write an item to the JSONL file"""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def generate_questions(doc_content, is_have_count):
    """Generate questions for a given document and write to JSONL"""
    # Message templates for question generation
    have_messages = [
        {
            "role": "user",
            "content": """
    다음 문서를 통해 대답할 수 있는 건강 정보 간결한 질문 예시 10개를 만들어주세요.:
    {{document}}

    json 형식의 리스트로 응답해주세요.
    ['질문1', '질문2', ... , '질문8']
    """
        }
    ]

    non_messages1 = [
        {
            "role": "user",
            "content": """
    다음 문서를 통해 대답할 수 있는 건강 정보 간결한 질문 예시 1개를 만들어주세요.:
    {{document}}

    json 형식의 리스트로 응답해주세요.
    ['질문1']
    """
        }
    ]
    messages = have_messages if is_have_count else non_messages1
    messages[0]['content'] = messages[0]['content'].replace('{{document}}', doc_content)
    try:
        # First attempt with Gemini
        gpt_response = client.chat.completions.create(
            model="google/gemini-flash-1.5-8b",
            messages=messages,
            temperature=0.6,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "json_object",
            }
        )
        
        response_content = gpt_response.choices[0].message.content
        print(response_content)
        questions = json.loads(response_content)
        
        # Write questions to JSONL
        for q in questions:
            item = {
                "question": q,
                "label": is_have_count,
            }
            write_to_jsonl('classify_dataset.jsonl', item)
        
        return questions
    
    except (json.JSONDecodeError, Exception):
        # Retry with alternative model
        try:
            print("Retrying with alternative model...")
            gpt_response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=messages,
                temperature=0.6,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                response_format={
                    "type": "json_object",
                },
                presence_penalty=0
            )
            
            response_content = gpt_response.choices[0].message.content
            questions = json.loads(response_content)
            
            # Write questions to JSONL
            for q in questions:
                item = {
                    "question": q,
                    "label": is_have_count,
                }
                write_to_jsonl('classify_dataset.jsonl', item)
            
            return questions
        except Exception as e:
            print(f"Failed to generate questions: {e}")
            return []

def process_chunk(chunk, is_have_count):
    """Process a single chunk and generate questions"""
    chunk_content = chunk['content']

    questions = generate_questions(chunk_content, is_have_count)
    return len(questions) if questions else 0

def main():
    # Read JSON file
    with open('./data/doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)
    docs.pop(0)
    
    have_count = 0
    non_count = 0
    total_questions = 0

    # Process each document sequentially
    for doc in docs:
        has_condition = False
        first_3_sentences = '. '.join(doc['content'].split('. ')[:3]) + '.'
        
        for condition in target_conditions:
            if condition in first_3_sentences:
                has_condition = True
                break
        
        if has_condition:
            have_count += len(doc['chunks'])
            for chunk in doc['chunks']:

                total_questions += process_chunk(chunk, True)
        else:
            non_count += len(doc['chunks'])
            for chunk in doc['chunks']:
                total_questions += process_chunk(chunk, False)

    print(f"Chunks in documents with conditions: {have_count}")
    print(f"Chunks in documents without conditions: {non_count}")
    print(f"Total questions generated: {total_questions}")

if __name__ == "__main__":
    main()