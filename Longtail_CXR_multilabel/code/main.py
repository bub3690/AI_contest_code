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

from src.datasets import MIMIC_CXR_Dataset
from src.utils import *
from src.losses import *


def main(args):
    # Set model/output directory name
    MODEL_NAME = args.dataset
    MODEL_NAME += f'_{args.model_name}'
    MODEL_NAME += f'_rand' if args.rand_init else ''
    MODEL_NAME += f'_decoupling-{args.decoupling_method}' if args.decoupling_method != '' else ''
    MODEL_NAME += f'_rw-{args.rw_method}' if args.rw_method != '' else ''
    MODEL_NAME += f'_{args.loss}'
    MODEL_NAME += '-drw' if args.drw else ''
    MODEL_NAME += f'_cb-beta-{args.cb_beta}' if args.rw_method == 'cb' else ''
    MODEL_NAME += f'_fl-gamma-{args.fl_gamma}' if args.loss == 'focal' else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_bs-{args.batch_size}'
    #현재 시간을 기준으로 모델 이름을 저장
    MODEL_NAME += f'_{time.strftime("%m%d%H%M%S")}'

    print(f'Model name: {MODEL_NAME}')
    # Create output directory for model (and delete if already exists)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    model_dir = os.path.join(args.out_dir, MODEL_NAME)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Create datasets + loaders
    dataset = MIMIC_CXR_Dataset
    N_CLASSES = 40
    
    label_df=pd.read_csv(os.path.join(args.label_dir, f'train_labeled.csv'))

    if args.check_phase:
        # data를 500개만 사용
        label_df = label_df[:10000]
    
    # train, valid split
    train_df, val_df = train_valid_split_custom(label_df, val_size=0.2, seed=args.seed)
    print(f'Train size: {len(train_df)}, Valid size: {len(val_df)}')

    train_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir,label_df=train_df, split='train')
    val_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir,label_df=val_df, split='valid')
    test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir,label_df=val_df, split='test')
    
    if args.decoupling_method == 'cRT':
        cls_weights = [len(train_dataset) / cls_count for cls_count in train_dataset.cls_num_list]
        
        # multilabel로 수정.
        instance_weights = []
        for labels in train_dataset.labels:
            label_weights = [cls_weights[label] for label in labels]  # 해당 인스턴스의 각 라벨에 대한 가중치
            instance_weight = sum(label_weights) / len(label_weights)  # 각 인스턴스의 평균 가중치
            instance_weights.append(instance_weight)
            
        sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(instance_weights), len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=val_worker_init_fn)

    # Create csv documenting training history
    history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'mAP', 'mAUC', 'mF1'])
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    # Set device
    device = torch.device('cuda:0')

    # Instantiate model
    #model = torchvision.models.resnet50(pretrained=(not args.rand_init))
    model = CNN('densenet121', num_classes=N_CLASSES)

    if args.decoupling_method == 'tau_norm':
        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')

        model.fc.bias.data = torch.zeros_like(model.fc.bias.data)
        fc_weights = model.fc.weight.data.clone()

        weight_norms = torch.norm(fc_weights, 2, 1)

        model.fc.weight.data = torch.stack([fc_weights[i] / torch.pow(weight_norms[i], -4) for i in range(N_CLASSES)], dim=0)
        
    elif args.decoupling_method == 'cRT':
        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')

        model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)  # re-initialize classifier head

    model = model.to(device)        

    # Set loss and weighting method
    if args.rw_method == 'sklearn':
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.labels), y=np.array(train_dataset.labels))
        weights = torch.Tensor(weights).to(device)
    elif args.rw_method == 'cb':
        weights = get_CB_weights(samples_per_cls=train_dataset.cls_num_list, beta=args.cb_beta)
        weights = torch.Tensor(weights).to(device)
    else:
        weights = None

    if weights is None:
        print('No class reweighting')
    else:
        print(f'Class weights with rw_method {args.rw_method}:')
        for i, c in enumerate(train_dataset.CLASSES):
            print(f'\t{c}: {weights[i]}')

    loss_fxn = get_loss(args, None if args.drw else weights, train_dataset)

    # Set optimizer
    if args.decoupling_method != '':
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start_time = time.time()  # 학습 시작 시간 기록
    
    
    # Train with early stopping
    if args.decoupling_method != 'tau_norm':
        epoch = 1
        early_stopping_dict = {'best_auc': 0.,'best_loss':9999.9, 'epochs_no_improve': 0}
        best_model_wts = deepcopy(model.state_dict())
        while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
            if args.decoupling_method == 'cRT':
                # froze backbone
                for param in model.parameters():
                    param.requires_grad = False
                
                for param in model.fc.parameters():
                    param.requires_grad = True
            
            history = train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES)
            history, early_stopping_dict, best_model_wts = validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, classes=val_dataset.CLASSES)

            if args.drw and epoch == 10:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1  # anneal LR
                loss_fxn = get_loss(args, weights, train_dataset)  # get class-weighted loss
                early_stopping_dict['epochs_no_improve'] = 0  # reset patience

            epoch += 1
    else:
        best_model_wts = model.state_dict()
        
    # Evaluate on imbalanced test set
    evaluate(args=args,model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts)
    end_time = time.time()  # 학습 종료 시간 기록
    epoch_duration = end_time - start_time  # 학습 시간 계산
        
    print(f"Train Finished in {epoch_duration/3600:.2f} hours.")




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
    
    
    # classifier retraining
    python main.py --data_dir /hdd/project/cxr_haziness/data/mimic/processed_images --num_workers 16 --out_dir mimic_results --dataset mimic-cxr-lt --loss bce --decoupling_method cRT --decoupling_weights /hdd/project/longtail/mimic_results/mimic-cxr-lt_densenet121_bce_lr-0.0001_bs-32-2/chkpt_epoch-13.pt --max_epochs 60 --patience 15 --batch_size 32 --lr 1e-4
    
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
    parser.add_argument('--decoupling_weights', type=str) # checkpoint
    parser.add_argument('--model_name', default='densenet121', type=str, help="CNN backbone to use")
    parser.add_argument('--max_epochs', default=60, type=int, help="maximum number of epochs to train")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=15, type=int, help="early stopping 'patience' during training")
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--n_TTA', default=0, type=int, help="number of augmented copies to use during test-time augmentation (TTA), default 0")
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
    parser.add_argument('--check_phase', action='store_true', default=False, help="check the phase of the code")
    
    args = parser.parse_args()

    print(args)

    main(args)