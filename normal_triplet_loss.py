from torch import nn
from torch.autograd import Variable
 
class TripletLoss(object):
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()
 
  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch 变量，anchor 和正样本之间的距离，shape [N]
      dist_an: pytorch 变量，anchor和负样本之间的距离，shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    # variable实现自动求导
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
      # y=1代表第一个输入的值，应该大于第二个输入的值即
    #   
    return loss