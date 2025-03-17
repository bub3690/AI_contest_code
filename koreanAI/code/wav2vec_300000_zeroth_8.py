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
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor

from datasets import load_dataset, load_metric, Audio


import torch
import torchaudio
from functools import partial

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union



from transformers import Trainer,TrainerCallback

from transformers import TrainingArguments

import re
from transformers import Wav2Vec2ForCTC

import nova
from nova import DATASET_PATH
from accelerate import notebook_launcher


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
        speech_data=resampler(speech_data)[0]
        speech_data = speech_data.to("cuda")
    
    speech_data = processor(audio=speech_data, sampling_rate=16000,return_tensors="pt").input_values[0]
    
    speech_data = speech_data.to("cuda")
    
    with torch.no_grad():
        logits = model(speech_data).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
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



def remove_special_characters(text):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\–\…]'
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text

    
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
    
    try:
        # hub_dataset["input_values"] = processor(audio=speech_data, sampling_rate=16000,return_tensors="pt").input_values[0]
        hub_dataset["input_values"] = processor(audio=speech_data, sampling_rate=16000,return_tensors="pt").input_values[0][0]
        
        hub_dataset["input_length"] = len(hub_dataset["input_values"])
        # print(hub_dataset["input_values"])
    except:
        print("inputvalue error : ",hub_dataset["filename"])
    try:
        hub_dataset["labels"] = processor(text=transcript).input_ids
    except:
        print("error : ",hub_dataset["filename"])
        print(transcript)
        
    return hub_dataset




# 데이터 패딩 시켜주는 함수
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        #label_features = [{"input_ids": feature["labels"]} for feature in features]
        #혹시 훈련데이터에 없는 문자열이 나오면, try해서 넘어가야함. None 제거.
        label_features = [{"input_ids": list( filter(None,feature["labels"]) ) } for feature in features] 
       
        # print(input_features[0].keys())
        # print(len(input_features))
        # print((len(input_features[0]["input_values"])))
        # print(len(input_features[0]["input_values"]))

        batch = self.processor.pad(
            input_features=input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )  
        
        # replace padding with [pad] to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels

        return batch



#정확도 계산
cer_metrics = load_metric("cer")
def compute_metrics(processor,pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metrics.compute(predictions=pred_str, references=label_str)

    # 5% 확률로 결과 출력
    if random.random() < 0.05:
        print("------------------")
        print("pred_str : ",pred_str)
        print("label_str : ",label_str)

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
                     "cer: {:.6f}, elapsed: {}, lr: {:.6f}"
                nova.report(
                    summary=True,
                    epoch=state.epoch,
                    train_loss=log_dict["loss"],
                    train_cer=0,
                    val_loss=log_dict["eval_loss"],
                    val_cer=log_dict["eval_cer"]
                )
                # print("report success")
                print(log_format.format(
                    current_timestamp, state.global_step, state.max_steps, log_dict["eval_loss"], log_dict["eval_cer"], elapsed, log_dict["learning_rate"]
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

        max_length = 300000
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
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                        sampling_rate=16000,
                                        padding_value=0.0,
                                        do_normalize=True,
                                        return_attention_mask=True)
    
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path,
                                    unk_token="<unk>",
                                    pad_token="<pad>",
                                    word_delimiter_token=" ")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)     
    print("processor make success")

    # model 제작
    print("model building")
    model = Wav2Vec2ForCTC.from_pretrained(
        #"facebook/wav2vec2-large-xlsr-53", # 이거 파일 받아와야함.
        "kresnik/wav2vec2-large-xlsr-korean", # zer
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )
    # CNN 파트는 학습 X
    model.freeze_feature_extractor()
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
                                               remove_columns=audio_dataset['train'].column_names, num_proc=8)
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

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
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
        
        training_args = TrainingArguments(
            output_dir='./results/',
            group_by_length=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2, #####
            evaluation_strategy="steps",
            num_train_epochs=200,
            gradient_checkpointing=True,
            fp16=True,
            save_steps=500000000,
            eval_steps=3094,#3125 대신
            logging_steps=3094,       
            learning_rate=3e-4, #####
            warmup_steps=500,
            save_total_limit=2,
            push_to_hub=False,
        )
        print(audio_dataset)
        print("parallel mode : ",training_args.parallel_mode)
        
        trainer = Trainer(
                model=model,
                data_collator=data_collator,
                args=training_args,
                #compute metrics with processor in parameter
                compute_metrics=partial(compute_metrics,processor),
                train_dataset=audio_dataset["train"],
                eval_dataset=audio_dataset["test"],
                tokenizer=processor.feature_extractor,
                # callbacks=[NovaCallback]
            )
        trainer.add_callback(NovaCallback(trainer))
        print("trainer build success")

        print("try to train")
        trainer.train()
        print("train success")

        print("try to remove callback")
        # trainer.remove_callback(NovaCallback)
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

