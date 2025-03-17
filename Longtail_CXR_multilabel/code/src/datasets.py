import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

class MIMIC_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir,label_df, split):
        self.split = split

        self.CLASSES = [
            'Adenopathy','Atelectasis','Azygos Lobe','Calcification of the Aorta','Cardiomegaly','Clavicle Fracture','Consolidation','Edema',
            'Emphysema','Enlarged Cardiomediastinum','Fibrosis','Fissure','Fracture','Granuloma','Hernia','Hydropneumothorax','Infarction','Infiltration',
            'Kyphosis','Lobar Atelectasis','Lung Lesion','Lung Opacity','Mass','Nodule','Normal','Pleural Effusion','Pleural Other','Pleural Thickening',
            'Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Pulmonary Embolism','Pulmonary Hypertension','Rib Fracture','Round(ed) Atelectasis',
            'Subcutaneous Emphysema','Support Devices','Tortuous Aorta','Tuberculosis'            
        ]
        
        
        self.label_df = label_df
        # label_df는 바깥에서 넘겨주는 방식으로 수정해야함. split 해야해서.
        
        self.img_paths = self.label_df['fpath'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()
        
        # 출력.
        print(f'Number of classes: {len(self.CLASSES)}')
        print(f'Number of samples: {len(self.img_paths)}')
        
        # class명, class별 sample 수 출력.
        for i, cls in enumerate(self.CLASSES):
            print(f'{cls}: {self.cls_num_list[i]}')
        print("--------------------")
        

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        #x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).float()


class MIMIC_CXR_Dataset_test(torch.utils.data.Dataset):
    def __init__(self, data_dir,label_df):

        self.CLASSES = [
            'Adenopathy','Atelectasis','Azygos Lobe','Calcification of the Aorta','Cardiomegaly','Clavicle Fracture','Consolidation','Edema',
            'Emphysema','Enlarged Cardiomediastinum','Fibrosis','Fissure','Fracture','Granuloma','Hernia','Hydropneumothorax','Infarction','Infiltration',
            'Kyphosis','Lobar Atelectasis','Lung Lesion','Lung Opacity','Mass','Nodule','Normal','Pleural Effusion','Pleural Other','Pleural Thickening',
            'Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Pulmonary Embolism','Pulmonary Hypertension','Rib Fracture','Round(ed) Atelectasis',
            'Subcutaneous Emphysema','Support Devices','Tortuous Aorta','Tuberculosis'            
        ]

        
        
        self.label_df = label_df

        self.img_paths = self.label_df['fpath'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        # path에서 .뒤에 제거,
        self.file_name = self.label_df['fpath'].apply(lambda x: x.split('.')[0]).values.tolist()


        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        #x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        return x.float(), self.file_name[idx]


## CREDIT TO https://github.com/agaldran/balanced_mixup ##

# pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches
    
    
    
if __name__ == '__main__':
    data_dir = '/hdd/project/cxr_haziness/data/mimic/processed_images'
    label_dir = '../labels/'
    label_df=pd.read_csv(os.path.join(label_dir, f'train_labeled.csv'))
    dataset = MIMIC_CXR_Dataset(data_dir, label_dir,label_df, 'train')
    print(dataset[0])
    
    