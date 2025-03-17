import random
import time
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from glob import glob
from tqdm import tqdm

# load pretrained model
### 수정
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

from datasets import load_dataset, load_metric, Audio


import torch
import torchaudio
from functools import partial



from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer



from transformers import Trainer,TrainerCallback
from transformers import TrainingArguments

import re
import time 

import nova
from nova import DATASET_PATH


def bind_model(model,processor, optimizer=None):
    def save(path, *args, **kwargs):
        torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
        #model.save_pretrained(path) #path: folder.(ddddd/ddd.ckpt)
        processor.save_pretrained(path)
        # print('Model saved',os.path.join(path, 'model.pt'))

    def load(path, *args, **kwargs):
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        #model.from_pretrained(path)
        processor.from_pretrained(path)
        model.to("cuda")
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model,processor)

    nova.bind(save=save, load=load, infer=infer)  # 'nova.bind' function must be called at the end.



def single_pred(model, processor, audio_path):
    speech_data, sampling_rate = torchaudio.load(audio_path)
    #resample
    if sampling_rate != 16000:
        #resample using torchaudio
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_data = resampler(speech_data)[0]
        #speech_data = speech_data.to("cuda")
    
    speech_data = speech_data[0]
    encoder_outputs = processor(audio=speech_data, sampling_rate=16000,return_tensors="pt")['input_features'].to("cuda")
    
    with torch.no_grad():
        logits = model.generate(encoder_outputs)

    transcription = processor.batch_decode(logits, skip_special_tokens=True)
    print(transcription)
    return transcription


def inference(path, model,processor, **kwargs):
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_pred(model,processor,i)[0] 
                # 
            }
        )
    return sorted(results, key=lambda x: x['filename'])
    
def speech_to_batch(hub_dataset,dataset_path,processor):
    # 데이터를 읽어서 배치로 만들기.
    
    # print(hub_dataset)
    # print(hub_dataset["filename"])

    audio_path = os.path.join(dataset_path, 'train', 'train_data',hub_dataset["filename"])
    
    speech_data, sampling_rate = torchaudio.load(audio_path)

    transcript = hub_dataset["text"]
    # print(transcript)

    #resample
    if sampling_rate != 16000:
        #resample using torchaudio
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_data=resampler(speech_data)[0]
    
        # hub_dataset["input_values"] = processor(audio=speech_data, sampling_rate=16000,return_tensors="pt").input_values[0]
    speech_data = speech_data[0]
    hub_dataset["input_features"] = processor.feature_extractor(speech_data, sampling_rate=16000,return_tensors="pt").input_features[0] # 0 하나 더 붙여야할 수도.
    ### 여기 오류날 수 있음. ***
    
    hub_dataset["labels"] = processor.tokenizer(text=transcript).input_ids
        
    return hub_dataset




# 데이터 패딩 시켜주는 함수
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



#정확도 계산
cer_metrics = load_metric("cer")
def compute_metrics(pred,processor):
    pred_logits = pred.predictions

    pred_str = processor.batch_decode(pred_logits,skip_special_tokens=True)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    cer = cer_metrics.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def convert_time(elapsed):
    day = time.strftime('%d', time.gmtime(elapsed))
    if day == '01':
        return time.strftime('%H:%M:%S', time.gmtime(elapsed))
    else:
        int_day = int(day) - 1
        return str(int_day)+ time.strftime(' day(s) :%H:%M:%S', time.gmtime(elapsed))

class NovaCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.begin_time = time.time()
        print("Starting training")

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        current_time = time.time()
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        log_format = "[{}] [INFO] step: {:4d}/{:4d}, elapsed: {}"
        
        elapsed = current_time - self.begin_time
        # trans elapsed time format
        elapsed_string = convert_time(elapsed) 
        
        log_dict = {}
        if state.log_history:
            log_dict = {}
            currnet_state = [log_dict.update(item) for item in state.log_history if item['step'] == state.global_step - 1]
            if log_dict:
                log_format = "[{}] [INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                     "cer: {:.6f}, elapsed: {}"
                nova.report(
                    summary=True,
                    epoch=state.epoch,
                    train_loss=log_dict["loss"],
                    train_cer=0,
                    val_loss=log_dict["eval_loss"],
                    val_cer=log_dict["eval_cer"]
                )
                print("report success")
                print(log_format.format(
                    current_timestamp, state.global_step, state.max_steps, log_dict["eval_loss"], log_dict["eval_cer"], elapsed
                ))
                print("model save : ",int(state.global_step))
                nova.save(int(state.global_step))

        if state.global_step % 10 == 0:
            print(log_format.format(
                current_timestamp, state.global_step, state.max_steps, elapsed_string
            ))

    # def on_epoch_end(self,args,state,control,logs=None,**kwargs):
    #     super().on_epoch_end(args, state, control, logs=None, **kwargs)
    #     print("model save : ",int(state.epoch))
    #     nova.save(int(state.global_step))
    #     print("model save success")
    #     self.epoch_begin_time = time.time()




def main(args):
    vocab_path = './vocab_v2.json'
    label_df_debug_path = './train_label_debug'
    if args.mode == 'train':    
        # DATASET_PATH = './data/Tr2DZ'
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        data_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_df = pd.read_csv(label_path) # transcript
        print(label_df.shape)
        
        max_length = 100000
        label_df_debug = label_df[:max_length] # debug
        # label_df_debug = label_df[:50] # debug
        label_df_debug.to_csv(label_df_debug_path, index=False)
        print("label_df_debug")
        print(label_df_debug)

        print("label_df")
        print(label_df)
        #filename, text 구조
    else:
        pass
    
    print("try to make processor")
    
    
    
    processor = WhisperProcessor.from_pretrained("seastar105/whisper-medium-ko-zeroth", language="Korean", task="transcribe")
    print("processor make success")

    # model 제작
    print("model 제작")
    model = WhisperForConditionalGeneration.from_pretrained("seastar105/whisper-medium-ko-zeroth")

    # CNN 파트는 학습 X
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print("model build success")

    print("try to bind model")
    # bind_model(model=model, processor=processor, vocab_dict=vocab_dict)
    bind_model(model=model, processor=processor)
    print("bind model success")

    if args.pause:    
        nova.paused(scope=locals())
        return
    

    print("try make dataset")
    # audio data load
    audio_dataset=load_dataset(
        "csv",
        data_files={
            # "train": label_path,
            "train": label_df_debug_path,
        },
    )
    
    # [filename,path,label,job,name]로 이뤄진 데이터
    audio_dataset = audio_dataset['train'].map(partial(speech_to_batch,dataset_path=DATASET_PATH,processor=processor),
                                               remove_columns=audio_dataset['train'].column_names, num_proc=16)
    print("dataset make success")
    print("try to split dataset")


    # train, valid split
    audio_dataset = audio_dataset.train_test_split(test_size=(2000 / max_length),seed=42) #일단은 랜덤하게 다 나눔.
    #추후에, 사람(name)을 겹치지 않게, 직업(job)을 비율에 맞춰서 나눠야함.
    print("split dataset success")


    print("try to make data collator")
    # max_length = max([len(x) for x in audio_dataset["train"]["input_values"] + audio_dataset["test"]["input_values"]])
    # max_length_labels = max([len(x) for x in audio_dataset["train"]["labels"] + audio_dataset["test"]["labels"]])
        # print("max_length : ",max_length)
        # print("max_length_labels : ",max_length_labels)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    #오류나는 곳
    print("make data collator success")
    
    ## Fine tuning

    print("try to save model")
    nova.save(0)
    print("save model success")
    
    # Augmentation 어디서 찾지?


    
    if args.mode == 'train':
        #이부분을 함수로 만들어서 distribute 호출하기.
        
        print("try to build trainer")
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results/",  # 원하는 리포지토리 이름을 임력한다.
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,  # 배치 크기가 2배 감소할 때마다 2배씩 증가
            learning_rate=1e-5,
            warmup_steps=500,
            num_train_epochs=200,
            gradient_checkpointing=True,
            fp16=True,
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            evaluation_strategy="steps",
            save_steps=500000000,
            eval_steps=5000,
            logging_steps=5000,
            load_best_model_at_end=False,
            metric_for_best_model="cer",  # 한국어의 경우 'wer'보다는 'cer'이 더 적합할 것
            greater_is_better=False,
        )        
        
        print(audio_dataset)
        print("parallel mode : ",training_args.parallel_mode)
        
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=audio_dataset["train"],
            eval_dataset=audio_dataset["test"],
            data_collator=data_collator,
            compute_metrics=partial(compute_metrics,processor=processor),
            tokenizer=processor.feature_extractor
        )
        trainer.add_callback(NovaCallback(trainer))
        print("trainer build success")

        print("try to train")
        trainer.train()
        print("train success")

        print("try to remove callback")
        trainer.remove_callback(NovaCallback)
        print("remove callback success")

        print("try to evaluate")
        trainer.evaluate()
        print("evaluate success")

    



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--pause', type=int, default=0)

    config = args.parse_args()
    print("train.py")
    main(config)
    #notebook_launcher(main, args=(config,),num_processes=2) ### 여기 수정.

    if config.pause:
        nova.paused(scope=locals())

