import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat

class MILoss(nn.Module):
    def __init__(self, temperature=10):
        super().__init__()
        print('=========using MI Loss=========')
        self.temperature = temperature
        self.eps = 1e-8

    def forward_feat2feat(self, feature_1, feature_2):
        """
        feauture_1 is enhanced image feature[batch, T, feat_dim]
        feature_2 is image feature[batch, T, feature_dim]
        """
        B, T, C = feature_2.shape
        feature_1 = feature_1.reshape(B*T, C) / self.temperature
        feature_2 = feature_2.reshape(B*T, C) / self.temperature

        log_softmax_1 = F.log_softmax(feature_1, dim=1)
        softmax_2 = F.softmax(feature_2, dim=1).clamp(min=self.eps)

        log_softmax_2 = F.log_softmax(feature_2, dim=1)
        softmax_1 = F.softmax(feature_1, dim=1).clamp(min=self.eps)

        loss = F.kl_div(log_softmax_1, softmax_2, reduction='batchmean') + F.kl_div(log_softmax_2, softmax_1)

        return loss

    def forward_label2feat(self, label, feature):
        """
        label is video label[batch, class]
        feature is image prediction[batch, T, class]
        """
        B, C = feature.shape
        label = label / self.temperature
        feature = feature / self.temperature

        log_softmax_feat = F.log_softmax(feature, dim=1)
        softmax_label = F.softmax(label*10, dim=1).clamp(min=self.eps)

        log_softmax_label = F.log_softmax(label*10, dim=1)
        softmax_feat = F.softmax(feature, dim=1).clamp(min=self.eps)

        loss = F.kl_div(log_softmax_feat, softmax_label, reduction='batchmean') + F.kl_div(log_softmax_label, softmax_feat,reduction='batchmean')

        return loss



if __name__ == '__main__':
    print('haha')
    feature_enhance = torch.randn(2, 8, 512)
    feature_simple = torch.randn(2, 8, 512)
    loss_mi = MILoss()
    feat_feat = loss_mi.forward_feat2feat(feature_enhance, feature_simple)

    label = torch.randn(2, 400)
    feature_prediction = torch.randn(2, 400)
    label_feat = loss_mi.forward_label2feat(label, feature_prediction)
    loss = feat_feat - label_feat
    print(loss)