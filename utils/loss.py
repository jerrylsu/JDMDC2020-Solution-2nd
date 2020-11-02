import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import torch
import torch.nn as nn


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1 - lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs * label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs * label, dim=1)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criterion = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)

    out = torch.randn(10, 5).cuda()
    lbs = torch.randint(5, (10,)).cuda()
    print('out:', out)
    print('lbs:', lbs)

    import torch.nn.functional as F

    loss = criterion(out, lbs)
    print('loss:', loss)