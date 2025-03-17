import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .models import model_dict


class CNN(nn.Module):
    def __init__(self,backbone,num_classes=40):
        super(CNN, self).__init__()
        self.backbone, self.num_features = model_dict[backbone]
        self.num_classes = num_classes
    
        # remove GAP, FC layer
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
        
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, embedding=False):        
        feature = self.backbone(x)
        feature = self.GAP(feature) # torch.Size([1, 6, 512, 16, 16])
        
        feature = self.fc(feature.view(-1, self.num_features))
        # feature가 아닌 logit을 남겨야하는 상황.
        
        return feature




if __name__ =="__main__":
    model = CNN('densenet121')
    print(model)
    print(model(torch.randn(1,3,512,512)).shape)
    
    
    
    


