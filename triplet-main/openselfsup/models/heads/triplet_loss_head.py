import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder

@HEADS.register_module
class TripletLossHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, gamma=2, size_average=True):
        super(TripletLossHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        #这个是什么呢？看后续是对输入的样本进行一个操作的。
        self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin=100.)#传统的那个triplet loss
        self.gamma = gamma

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        #这里的输入是什么呢？处理前后？是一个batch还是只是一个正负样本？看起来input是一个batch的正负样本
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        # 对输入的数据（tensor）进行指定维度的L2_norm运算。
        #L2归一化：将一组数变成0-1之间。
        n = input.size(0)
        #输入的样本的数量吗？
        dist = -2. * torch.matmul(pred_norm, target_norm.t())
        idx = torch.arange(n)
        #生成序号
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())
        #相当于生成一个n*n大小的，对角线都为一其余都为0的矩阵
        dist_ap, dist_an = [], []
        #ap值与正样本的距离，an指与负样本距离
        for i in range(n):
            '''
            作用是用来计算一个minibatch中的正负。
            根据mask的值，可以知道dist的对角线是作为正样本的距离的
            我猜测是因为pred_norm和target_norm对角线的位置分贝是正样本与锚定样本?
            '''
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            #计算正样本，相当于选出dist[i]中与mask[i]中为1的值同位置的值。可以详细的看一下学院服务器的内容。
            down_k, _ = torch.topk(dist[i][mask[i]==0], 10, dim=-1, largest=False)
            #dist[i][mask[i]==0]，选出mask中为0的位置对应的一系列结果
            down_k = down_k[1:].mean().unsqueeze(0)
            #选取前k个求均值，作为负样本，为什么是从1开始呢？
            dist_an.append(down_k)
            # dist_an.append(dist[i][mask[i] == 0].median().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)
        return dict(loss=loss_triplet)
            
