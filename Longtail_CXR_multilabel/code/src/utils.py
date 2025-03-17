import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from copy import deepcopy

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)




def train_valid_split_custom(label_df, val_size, seed):
    class_columns = [
            'Adenopathy','Atelectasis','Azygos Lobe','Calcification of the Aorta','Cardiomegaly','Clavicle Fracture','Consolidation','Edema',
            'Emphysema','Enlarged Cardiomediastinum','Fibrosis','Fissure','Fracture','Granuloma','Hernia','Hydropneumothorax','Infarction','Infiltration',
            'Kyphosis','Lobar Atelectasis','Lung Lesion','Lung Opacity','Mass','Nodule','Normal','Pleural Effusion','Pleural Other','Pleural Thickening',
            'Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Pulmonary Embolism','Pulmonary Hypertension','Rib Fracture','Round(ed) Atelectasis',
            'Subcutaneous Emphysema','Support Devices','Tortuous Aorta','Tuberculosis'
    ]

    # 다중 클래스 라벨을 하나의 라벨로 결합 (예: 튜플 형태)
    label_df['combined_labels'] = list(label_df[class_columns].itertuples(index=False, name=None))

    # 각 클래스의 샘플 수 계산
    label_counts = label_df['combined_labels'].value_counts()

    # 클래스 라벨 수가 1개인 샘플 분리
    single_class_samples = label_df[label_df['combined_labels'].isin(label_counts[label_counts == 1].index)]
    multiple_class_samples = label_df[label_df['combined_labels'].isin(label_counts[label_counts > 1].index)]

    # 다중 클래스 샘플을 stratify하여 훈련 세트와 검증 세트로 분할
    train_multiple, val_multiple = train_test_split(
        multiple_class_samples, test_size=val_size, random_state=seed, stratify=multiple_class_samples['combined_labels']
    )

    # 단일 클래스 샘플을 train set에 추가 (val set으로 들어가지 않도록 함)
    train_df = pd.concat([train_multiple, single_class_samples]).reset_index(drop=True)
    val_df = val_multiple.reset_index(drop=True)

    # 불필요한 열 제거
    train_df = train_df.drop(columns=['combined_labels'])
    val_df = val_df.drop(columns=['combined_labels'])

    return train_df, val_df




def train(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, classes):
    """Train PyTorch model for one epoch on NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    running_loss = 0.
    y_true, y_hat = [], []
    for i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)


        out = model(x)

        loss = loss_fxn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        out = torch.sigmoid(out)
        y_hat.append(out.detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics for multilabel classification
    try:
        mAP = average_precision_score(y_true, y_hat, average='macro')
    except:
        print("average_precision_score error")
        mAP = 0.0
    
    try:
        mAUC = roc_auc_score(y_true, y_hat, average='macro')
    except:
        print("roc_auc_score error")
        mAUC = 0.0
    
    try:
        mF1 = f1_score(y_true, (y_hat > 0.5).astype(int), average='macro')
    except:
        print("f1_score error")
        mF1 = 0.0

    print('mAP:', round(mAP, 3), '|', 'mAUC:', round(mAUC, 3), '|', 'mF1:', round(mF1, 3))

    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (i + 1), mAP, mAUC, mF1]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return pd.concat([history, current_metrics], ignore_index=True)


def validate(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, classes):
    """Evaluate PyTorch model on validation set.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        early_stopping_dict : dict
            Dictionary of form {'epochs_no_improve': <int>, 'best_loss': <float>} for early stopping
        best_model_wts : PyTorch state_dict
            Model weights from best epoch
        classes : list[str]
            Ordered list of names of output classes
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed validation epoch
        early_stopping_dict : dict
            Updated early stopping metrics
        best_model_wts : PyTorch state_dict
            (Potentially) updated model weights (if best validation loss achieved)
    """
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')

    running_loss = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = loss_fxn(out, y)

            running_loss += loss.item()

            out = torch.sigmoid(out)
            y_hat.append(out.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics for multilabel classification
    try:
        mAP = average_precision_score(y_true, y_hat, average='macro')
    except:
        print("average_precision_score error")
        mAP = 0.0
    
    try:
        mAUC = roc_auc_score(y_true, y_hat, average='macro')
    except:
        print("roc_auc_score error")
        mAUC = 0.0
    
    try:
        mF1 = f1_score(y_true, (y_hat > 0.5).astype(int), average='macro')
    except:
        print("f1_score error")
        mF1 = 0.0

    print('[VAL] mAP:', round(mAP, 3), '|', 'mAUC:', round(mAUC, 3), '|', 'mF1:', round(mF1, 3))

    current_metrics = pd.DataFrame([[epoch, 'val', running_loss / (i + 1), mAP, mAUC, mF1]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # Early stopping: save model weights only when val loss has improved
    if mAUC > early_stopping_dict['best_auc']:
        print(f'--- EARLY STOPPING: Validation AUC has improved from {round(early_stopping_dict["best_auc"], 3)} to {round(mAUC, 3)}! Saving weights. ---')
        early_stopping_dict['epochs_no_improve'] = 0
        early_stopping_dict['best_auc'] = mAUC
        best_model_wts = deepcopy(model.state_dict())
        torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
    else:
        print(f'--- EARLY STOPPING: Validation AUC has not improved from {round(early_stopping_dict["best_auc"], 3)} ---')
        early_stopping_dict['epochs_no_improve'] += 1

    return pd.concat([history, current_metrics], ignore_index=True), early_stopping_dict, best_model_wts



def evaluate(args, model, device, loss_fxn, dataset, split, batch_size, history, model_dir, weights):
    """Evaluate PyTorch model on test set of NIH ChestXRay14 dataset. Saves training history csv, summary text file, training curves, etc.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        batch_size : int
        history : pandas DataFrame
            Data frame containing history of training metrics
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        weights : PyTorch state_dict
            Model weights from best epoch
        n_TTA : int
            Number of augmented copies to use for test-time augmentation (0-K)
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    """
    model.load_state_dict(weights)  # load best weights
    model.eval()

    # INFERENCE
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers if split == 'test' else 2, pin_memory=True)

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')

    running_loss = 0.
    y_true, y_hat = [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = loss_fxn(out, y)

            running_loss += loss.item()

            out = torch.sigmoid(out)
            y_hat.append(out.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    # Compute metrics for multilabel classification
    try:
        mAP = average_precision_score(y_true, y_hat, average='macro')
    except:
        print("average_precision_score error")
        mAP = 0.0
    
    try:
        mAUC = roc_auc_score(y_true, y_hat, average='macro')
    except:
        print("roc_auc_score error")
        mAUC = 0.0
    
    try:
        mF1 = f1_score(y_true, (y_hat > 0.5).astype(int), average='macro')
    except:
        print("f1_score error")
        mF1 = 0.0

    print(f'[{split.upper()}] mAP: {round(mAP, 3)} | mAUC: {round(mAUC, 3)} | mF1: {round(mF1, 3)}')

    # Save true and predicted labels
    pred_df = pd.DataFrame(y_hat, columns=dataset.CLASSES)
    true_df = pd.DataFrame(y_true, columns=dataset.CLASSES)

    pred_df.to_csv(os.path.join(model_dir, f'{split}_pred.csv'), index=False)
    true_df.to_csv(os.path.join(model_dir, f'{split}_true.csv'), index=False)

    # Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'loss.png'), dpi=300, bbox_inches='tight')

    # Plot mAP curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'mAP'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'mAP'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'mAP.png'), dpi=300, bbox_inches='tight')

    # Plot mAUC curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'mAUC'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'mAUC'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAUC')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'mAUC.png'), dpi=300, bbox_inches='tight')

    # Plot mF1 curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'mF1'], label='train')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'mF1'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mF1')
    ax.legend()
    fig.savefig(os.path.join(model_dir, 'mF1.png'), dpi=300, bbox_inches='tight')

    # Create summary text file describing final performance
    summary = f'mAP: {round(mAP, 3)}\n'
    summary += f'mAUC: {round(mAUC, 3)}\n'
    summary += f'mF1: {round(mF1, 3)}\n\n'

    cls_report = classification_report(y_true, (y_hat > 0.5).astype(int), target_names=dataset.CLASSES, digits=3)
    summary += cls_report

    with open(os.path.join(model_dir, f'{split}_summary.txt'), 'w') as f:
        f.write(summary)
        
        
        
def Inference(args, model, device, data_loader):
    print("Inference")
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Inference')

    y_hat = []
    img_path_list = []
    with torch.no_grad():
        for i, (x,img_path) in pbar:
            x = x.to(device)

            out = model(x)

            out = torch.sigmoid(out)
            y_hat.append(out.detach().cpu().numpy())
            img_path_list.append(img_path)

    # Collect true and predicted labels into flat numpy arrays
    y_hat = np.concatenate(y_hat)
    img_path_list = np.concatenate(img_path_list)

    return y_hat, img_path_list

        
        
