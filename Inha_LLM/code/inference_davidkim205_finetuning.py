
import os
import json
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, PeftConfig



peft_model_id = "yujinjin/david-gemma-finetuned-4400" ##모델 이름 수정
config = PeftConfig.from_pretrained(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
)
model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
)


def generate_response(context, question, model, tokenizer):
    prompt = f"""너는 주어진 문맥을 토대로 질문에 대해 간결하고 정확하게 답변하는 챗봇이야.
답변은 반드시 문맥 내의 표현과 일치해야 하며, 문맥에서 직접 인용한 정보를 기반으로 작성해야 해.

**답변 가이드라인:**
1. **정확성**: 질문에 대한 답변은 문맥 내의 텍스트에서 직접 인용하여 작성하며, 문맥의 표현과 반드시 일치해야 합니다.
2. **간결성**: 답변은 문맥 내의 단어와 구절을 그대로 사용하며, 불필요한 정보를 생략하세요.
3. **핵심 정보**: 답변에는 필요 최소한의 정보만 포함하며, 문맥에서 제공된 정확한 표현을 사용하세요.
4. **줄임말 사용 금지**: 이름, 회사명, 지역명, 정책명 등은 줄임말로 답변하지 말고 전체 이름으로 답변하세요.

**예시 질문/답변:**

- 질문: 체납자의 가상화폐를 강제 처분하기 위해 회원 정보를 어디에 요청했어
 - 답변: 업비트, 코인원, 빗썸, 코빗

- 질문: 건설사업 장소의 특별검사를 통해 한국농어촌공사가 어떤 걸 대응하려는 거야
 - 답변: 여름철 폭염과 집중호우

- 질문: 애플사의 무슨 행위가 현대차그룹의 주가를 폭등시킨 거야
 - 답변: 애플카 출시를 위해 현대차에 협력을 제안

- 질문: 블루오벌에스케이에서 무슨 차량 부품을 생산하려고 해
 - 답변: 60GWh 규모의 전기차 배터리

- 질문: 비지비즈가 제시하는 영어 표현 가이드북의 목적은
 - 답변: 대화유도

- 질문: 소규모 제조업체들이 경영에 어려움을 겪는 주된 이유
 - 답변: 물류비용

- 질문: 신선한 과일을 고를 때 사과의 경우 무엇을 보면 돼
 - 답변: 향과 꼭지

- 질문: 어떤 부분이 수소 사업에 대해 사용자들이 우려를 표하는 거야
 - 답변: 수소 안전 문제

- 질문: 언제부터 과학기술정보통신부가 인공지능 융합 부문의 신규과제를 모집해
 - 답변: 3일

- 질문: 비스쿡 전기그릴을 쇼핑 방송에서 볼 수 있는 건 언제야
 - 답변: 2일 오후 7시

- 질문: 16일에 복수의결권 도입을 밀어붙이고 있다고 누가 말했어
 - 답변: 권칠승

**답변 작성 지침:**
1. 답변은 문맥에서 언급된 내용과 표현이 정확히 일치해야 합니다.
2. 인물, 날짜, 장소, 나이, 지역을 묻는 질문일 경우 한 단어로 답변하세요.
3. 이름, 회사명, 지역명, 정책명 등은 줄임말로 답변하지 말고 전체 이름으로 답변하세요.
4. **질문 유형에 따른 답변 처리**:
   - **하나만 답하라는 질문**: 문맥에서 가장 직접적이고 관련된 하나의 답변을 선택하여 제공하세요.
   - **두 개를 답하라는 질문**: 문맥에서 언급된 모든 관련된 답변 항목을 포함하여 제공하세요.
5. **중복 답변 처리**:
   - 문맥에서 중복된 답변이 있는 경우, 중복된 정보를 필터링하여 가장 관련성 높은 정보를 선택하세요.
   - 모든 중복된 정보를 제공할 필요가 없으며, 문맥에 맞는 정확한 답변을 제공하세요.
6. 답변에 대한 부가 설명은 제외하고, 필요한 핵심 정보만 포함하세요.
7. 답변은 문맥의 표현과 반드시 일치해야 하며, 최대 5단어 이내의 핵심 키워드로 답변하세요.

다음 문맥을 정확하게 이해해줘.

**문맥:** {context}

문맥에 대한 질문이야.

**질문:** {question}?

답변은 위 7가지 조건을 만족해야 하며, 만족하지 않으면 다시 작성해야 합니다.

답변::"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=70, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=0.24, top_p=0.9, top_k=50,do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "답변::" in response:
        response = response.split("답변::", 1)[1][:40]
        if "질문" in response:
            response = response.split("질문", 1)[0]
        if "옳은" in response:
            response = response.split("옳은", 1)[0]
        if "다음" in response:
            response = response.split("다음", 1)[0]
        if "이렇게" in response:
            response = response.split("이렇게", 1)[0]
        if "문맥" in response:
            response = response.split("문맥", 1)[0]
        if "답변" in response:
            response = response.split("답변", 1)[0]
        if "다른" in response:
            response = response.split("다른", 1)[0]
        if "키워드" in response:
            response = response.split("키워드", 1)[0]
        if "이유" in response:
            response = response.split("이유", 1)[0]
        if "틀린 답변" in response:
            response = response.split("틀린 답변", 1)[0]
        if "추가 설명" in response:
            response = response.split("추가 설명", 1)[0]
        if "*" in response:
            response = response.replace("*", "")
        if "부터" in response:
            response = response.replace("부터", "")
        if "\n" in response:
            response = response.replace("\n", "")
    return response.strip()

def remove_newlines(text):
    return text.replace('\n', '')

def remove_double(text):
    return text.replace('  ', '')

def remove_noise(text):
    return text.replace('#', '')

def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        '''연속된 공백일 경우 하나의 공백으로 대체'''
        return ' '.join(text.split())

    def remove_punc(text):
        '''구두점 제거'''
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        '''소문자 전환'''
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # 문자 단위로 f1-score를 계산 합니다.
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def evaluate(ground_truth_df, predictions_df):
    predictions = dict(zip(predictions_df['question'], predictions_df['answer']))
    f1 = exact_match = total = 0

    for index, row in ground_truth_df.iterrows():
        question_text = row['question']
        ground_truths = row['answer']
        total += 1
        if question_text not in predictions:
            continue
        prediction = predictions[question_text]
        f1 = f1 + f1_score(prediction, ground_truths)

    f1 = 100.0 * f1 / total
    return {'f1': f1}

def main():
    test_data = pd.read_csv('./test.csv')
    test_data['context'] = test_data['context'].apply(remove_newlines)
    test_data['context'] = test_data['context'].apply(remove_double)
    test_data['context'] = test_data['context'].apply(remove_noise)

    submission_dict = {}

    for index, row in test_data.iterrows():
        try:
            context = row['context']
            question = row['question']
            id = row['id']

            if context is not None and question is not None:
                answer = generate_response(context, question, model, tokenizer)
                submission_dict[id] = answer

                # 질문과답변 출력
                print(f"id: {id}")
                print(f"질문: {question}")
                print(f"답변: {answer}")
                print("-" * 50)
            else:
                submission_dict[id] = 'Invalid question or context'
                print(f"Invalid question or context for id: {id}")
                print("-" * 50)

        except Exception as e:
            print(f"Error processing question {id}: {e}")
            print("-" * 50)

    df = pd.DataFrame(list(submission_dict.items()), columns=['id', 'answer'])
    df.to_csv( './davidkim205_final_v4.csv', index=False)

main()
