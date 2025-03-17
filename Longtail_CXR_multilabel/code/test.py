import os
import shutil

import argparse
import numpy as np
import pandas as pd
import torch
import torchvision

from sklearn.utils import class_weight

from src.model import CNN
import time

from src.datasets import MIMIC_CXR_Dataset_test
from src.utils import *
from src.losses import *


def main(args):
    # Set model/output directory name

    # Create output directory for model (and delete if already exists)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Create datasets + loaders
    dataset = MIMIC_CXR_Dataset_test
    N_CLASSES = 40
    
    test_df = pd.read_csv(os.path.join(args.label_dir, f'development.csv'))
    
    if args.check_phase:
        test_df = test_df[0:100]
        
    test_dataset = dataset(data_dir=args.data_dir, label_df=test_df)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Set device
    device = torch.device('cuda:0')

    # Instantiate model
    #model = torchvision.models.resnet50(pretrained=(not args.rand_init))
    model = CNN('densenet121', num_classes=N_CLASSES)

    # load checkpoint
    model.load_state_dict(torch.load(args.checkpoint)['weights'])
    model = model.to(device)
    
    print("Model loaded from checkpoint")
    
    # Evaluate on imbalanced test set
    start_time = time.time()  # 학습 시작 시간 기록
    y_hat, img_path_list = Inference(args=args, model=model, device=device, data_loader=test_data_loader)
    #evaluate(args=args,model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts)
    
    end_time = time.time()  # 학습 종료 시간 기록
    epoch_duration = end_time - start_time  # 학습 시간 계산
    
    
    
    # EVAL Class list
    train_cls = [
            'Adenopathy','Atelectasis','Azygos Lobe','Calcification of the Aorta','Cardiomegaly','Clavicle Fracture','Consolidation','Edema',
            'Emphysema','Enlarged Cardiomediastinum','Fibrosis','Fissure','Fracture','Granuloma','Hernia','Hydropneumothorax','Infarction','Infiltration',
            'Kyphosis','Lobar Atelectasis','Lung Lesion','Lung Opacity','Mass','Nodule','Normal','Pleural Effusion','Pleural Other','Pleural Thickening',
            'Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Pulmonary Embolism','Pulmonary Hypertension','Rib Fracture','Round(ed) Atelectasis',
            'Subcutaneous Emphysema','Support Devices','Tortuous Aorta','Tuberculosis'            
        ]
    
    eval_cls = ["Atelectasis", "Calcification of the Aorta","Cardiomegaly","Consolidation","Edema","Emphysema","Enlarged Cardiomediastinum","Fibrosis","Fracture","Hernia","Infiltration","Lung Lesion","Lung Opacity",
                "Mass","Normal","Nodule","Pleural Effusion","Pleural Other","Pleural Thickening","Pneumomediastinum","Pneumonia","Pneumoperitoneum","Pneumothorax","Subcutaneous Emphysema","Support Devices","Tortuous Aorta",
                ]
    
    
    # eval_cls에 맞도록 y_hat을 수정하고 csv 저장
    
    # sample_submission = pd.read_csv(os.path.join(args.label_dir, 'sample_submission.csv'))
    # # row들을 제거.
    # sample_submission = sample_submission[0:0]
    # dicom_id	Atelectasis	Calcification of the Aorta	Cardiomegaly	Consolidation	Edema	Emphysema	Enlarged Cardiomediastinum	Fibrosis	Fracture	...	Pleural Effusion	Pleural Other	Pleural Thickening	Pneumomediastinum	Pneumonia	Pneumoperitoneum	Pneumothorax	Subcutaneous Emphysema	Support Devices	Tortuous Aorta
    # y_hat
    dicom_id = img_path_list
    
    
    # dicom_id with y_hat
    submit_df = pd.DataFrame(dicom_id, columns=['dicom_id'])
    cls_df = pd.DataFrame(y_hat, columns=train_cls)

     
    # class를 eval_cls에 맞도록 수정.
    cls_df = cls_df[eval_cls]
    
    # dicom_id with y_hat
    submit_df = pd.concat([submit_df, cls_df], axis=1)    
    
    # csv 저장
    submit_df.to_csv(os.path.join(args.out_dir, 'submission.csv'), index=False)
    
    
    

if __name__ == '__main__':
    # Command-line arguments
    """
    python main.py --data_dir /hdd/project/cxr_haziness/data/mimic/processed_images \
                     --out_dir mimic_results \
                     --dataset mimic-cxr-lt \
                     --loss ldam \
                     --max_epochs 60 \
                     --patience 15 \
                     --batch_size 256 \
                     --lr 1e-4 \
                         
    python main.py --data_dir /hdd/project/cxr_haziness/data/mimic/processed_images --num_workers 16 --out_dir mimic_results --dataset mimic-cxr-lt --loss ldam --max_epochs 60 --patience 15 --batch_size 32 --lr 1e-4
    python main.py --data_dir /hdd/project/cxr_haziness/data/mimic/processed_images --num_workers 16 --out_dir mimic_results --dataset mimic-cxr-lt --loss ldam --max_epochs 2 --patience 15 --batch_size 32 --lr 1e-4
    
    python main.py --data_dir /shared/home/mai/jongbub/data/mimic/processed_images --num_workers 10 --out_dir mimic_results --dataset mimic-cxr-lt --loss ldam --max_epochs 60 --patience 15 --batch_size 64 --lr 1e-4
    
    python main.py --data_dir /hdd/project/cxr_haziness/data/mimic/processed_images --num_workers 16 --out_dir mimic_results --dataset mimic-cxr-lt --loss bce --max_epochs 60 --patience 15 --batch_size 32 --lr 1e-4
    
    
    
    
    # test code
    python test.py --data_dir /hdd/project/cxr_haziness/data/mimic/processed_images --label_dir ./labels/ --out_dir results/ --dataset mimic-cxr-lt --num_workers 16 --check_phase
    python test.py --data_dir /hdd/project/cxr_haziness/data/mimic/processed_images --label_dir ./labels/ --out_dir results/ --dataset mimic-cxr-lt --num_workers 8 --checkpoint /hdd/project/longtail/mimic_results/mimic-cxr-lt_densenet121_decoupling-cRT_bce_lr-0.0001_bs-64_0803193402/chkpt_epoch-60.pt
    
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/shared/home/mai/jongbub/data/mimic/processed_images', type=str)
    parser.add_argument('--label_dir', default='./labels/', type=str)
    parser.add_argument('--out_dir', default='results/', type=str, help="path to directory where results and model weights will be saved")
    parser.add_argument('--dataset', required=True, type=str, choices=['nih-lt', 'mimic-cxr-lt'])
    parser.add_argument('--num_workers', default=0, type=int, help="number of workers for data loading")
    parser.add_argument('--loss', default='ce', type=str, choices=['bce', 'focal', 'ldam'])
    parser.add_argument('--drw', action='store_true', default=False)
    parser.add_argument('--rw_method', default='', choices=['', 'sklearn', 'cb'])
    parser.add_argument('--cb_beta', default=0.9999, type=float)
    parser.add_argument('--fl_gamma', default=2., type=float)
    parser.add_argument('--bal_mixup', action='store_true', default=False)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', default=0.2, type=float)
    parser.add_argument('--decoupling_method', default='', choices=['', 'cRT', 'tau_norm'], type=str)
    parser.add_argument('--decoupling_weights', type=str)
    parser.add_argument('--model_name', default='densenet121', type=str, help="CNN backbone to use")
    parser.add_argument('--max_epochs', default=60, type=int, help="maximum number of epochs to train")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=15, type=int, help="early stopping 'patience' during training")
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--n_TTA', default=0, type=int, help="number of augmented copies to use during test-time augmentation (TTA), default 0")
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
    
    parser.add_argument('--check_phase', action='store_true', default=False)
    parser.add_argument('--checkpoint', default="/hdd/project/longtail/mimic_results/mimic-cxr-lt_densenet121_bce_lr-0.0001_bs-32/chkpt_epoch-13.pt", type=str)
    
    

    args = parser.parse_args()

    print(args)

    main(args)