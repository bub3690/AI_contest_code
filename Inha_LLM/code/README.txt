파일 설명
<Inference 코드>
1. inference_davidkim205_finetuning 
-> "yujinjin/david-gemma-finetuned-4400" 모델 불러와서 모델 inference 진행
-> temperature=0.6, top_p=0.95, top_k=50 
유의 : generate 과정 argument 값이 높아 inference 할때마다 output 값이 달라짐

2. finetuned_inference
-> "yujinjin/david-gemma-finetuned-4400" 모델 불러와서 모델 inference 진행
-> temperature=0.3

3. david_finetune_ver3_inference
-> "yujinjin/david-gemma-finetuned-4400" 모델 불러와서 모델 inference 진행
-> temperature=0.6, top_p=0.95, top_k=50,
유의 : generate 과정 argument 값이 높아 inference 할때마다 output 값이 달라짐


<Ensemble>
1. LLM 앙상블 방식 중 Voting 방식을 활용
-> inference에서 생성된 3개의 답변을 비교해 2개이상 같은 값은 남기고 만약 3개의 답변 모두 다를때는 가장 짧은 답변만 남긴다.



<FineTuning>
ko_gemma_qlora_fintune
-> davidkim205/ko-gemma-2-9b-it 모델 qlora 방식으로 Finetuning
-> checkpoint 허깅페이스에 업로드 ("yujinjin/david-gemma-finetuned-4400")
*2024.08.14(수) 20시 59분 기준 Private Checkpoint 허깅 페이스에 업로드된 상태
*2024.08.14(수) 22시전까지 Public 수정 예정 & 허깅페이스에서 확인 가능
(용량이 커 checkpoint 첨부는 생략했습니다 -> checkpoint weight 등은 허깅페이스에서 확인 가능)

개발 환경
liux 리눅스:  (RTX A40)

라이브러리 버전
numpy : 1.24.1 / pandas : 2.0.3 / tqdm : 4.66.4 / peft : 0.12.0 / torch : 2.1.0+cu118 / transformers : 4.44.0 / datasets : 2.22.0 / huggingface_hub : 0.24.5 / accelerate :  0.33.0 / trl : 0.9.6