import numpy as np
import torch
import torch.nn.functional as F

def get_loss(args, weights, train_dataset):
    if args.loss == 'bce':
        loss_fxn = torch.nn.BCEWithLogitsLoss(weight=weights, reduction='mean')
    elif args.loss == 'focal':
        loss_fxn = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=weights, gamma=args.fl_gamma, reduction='mean')
    elif args.loss == 'ldam':
        loss_fxn = MultiLabelLDAMLoss(cls_num_list=train_dataset.cls_num_list, weight=weights)
    elif args.loss == 'bs':
        loss_fxn = BalancedSoftmaxLoss_multilabel(sample_per_class=train_dataset.cls_num_list, reduction='mean')

    return loss_fxn


def get_CB_weights(samples_per_cls, beta):
    effective_num = 1.0 - np.power(beta, samples_per_cls)

    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(samples_per_cls)

    return weights


## CREDIT TO https://github.com/kaidic/LDAM-DRW ##
class MultiLabelLDAMLoss(torch.nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(MultiLabelLDAMLoss, self).__init__()
        cls_num_list = [1 if x == 0 else x for x in cls_num_list]
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = target.float()
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(index_float, self.m_list[None, :].transpose(0, 1))
        x_m = x - batch_m
        
        # BCEWithLogitsLoss 사용
        loss = F.binary_cross_entropy_with_logits(self.s * x_m, target.float(), weight=self.weight, reduction='none')
        
        # 각 샘플의 각 클래스 손실을 합산
        #loss = loss.mean(dim=1)
        
        # 배치 전체의 평균 손실 계산
        return loss.mean()


class BalancedSoftmaxLoss_multilabel(torch.nn.Module):
    def __init__(self, sample_per_class, reduction='mean'):
        """
        Initialize the Balanced Softmax Loss module for multi-label classification.
        Args:
            sample_per_class: A tensor of size [no_of_classes], the number of samples per class.
            reduction: string, one of "none", "mean", "sum". Determines the reduction to apply to the loss.
        """
        super(BalancedSoftmaxLoss_multilabel, self).__init__()
        self.sample_per_class = sample_per_class
        self.reduction = reduction
        
    def forward(self, logits, labels):
        """
        Forward pass to compute the Balanced Softmax Loss for multi-label classification.
        Args:
            logits: A float tensor of size [batch, no_of_classes], predicted logits.
            labels: A float tensor of size [batch, no_of_classes], ground truth binary labels.
        Returns:
            loss: A float tensor, the computed Balanced Softmax Loss.
        """
        # Convert sample_per_class to the same type as logits and adjust dimensions
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand_as(logits)
        
        # Adjust logits based on the log of the sample count per class
        adjusted_logits = logits + spc.log()
        
        # Compute the binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(adjusted_logits, labels, reduction=self.reduction)
        
        # Return the calculated loss
        return bce_loss
    
    