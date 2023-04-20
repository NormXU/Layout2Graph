import copy
import torch
import torch.nn.functional as F
from torch.nn import CTCLoss, MSELoss, KLDivLoss, L1Loss, SmoothL1Loss
from torch.nn.functional import log_softmax
import numpy as np


# 梯度scale
def norm_grad(inputs_adv, inputs_adv_grad, norm_type, epsilon=0.3, norm_epsilon=1e-6, alpha=1e-5, inputs_origin=None):
    if norm_type == 'l2':
        inputs_adv = inputs_adv + alpha * inputs_adv_grad / (torch.norm(inputs_adv_grad, dim=-1, keepdim=True) +
                                                             norm_epsilon)
    elif norm_type == 'sign':
        inputs_adv = inputs_adv + alpha * torch.sign(inputs_adv_grad.data)
        inputs_adv = torch.clip(inputs_adv, 0, 1)
    elif norm_type == 'sign_pgd':
        inputs_adv = inputs_adv + alpha * torch.sign(inputs_adv_grad.data)
        inputs_adv = torch.clip(inputs_adv, inputs_origin - epsilon, inputs_origin + epsilon)
        inputs_adv = torch.clip(inputs_adv, 0, 1)
    elif norm_type == 'l2_pgd':
        inputs_adv = inputs_adv + alpha * inputs_adv_grad / (torch.norm(inputs_adv_grad, dim=-1, keepdim=True) +
                                                             norm_epsilon)
        r = inputs_adv - inputs_origin
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
            inputs_adv = inputs_origin + r
    elif norm_type == 'sign_freelb':
        inputs_adv_tmp = inputs_adv + alpha * torch.sign(inputs_adv_grad.data)
        inputs_adv_tmp = torch.clip(inputs_adv_tmp, inputs_adv - epsilon, inputs_adv + epsilon)
        inputs_adv = torch.clip(inputs_adv_tmp, 0, 1)
    elif norm_type == 'l2_freelab':
        inputs_adv_tmp = inputs_adv + alpha * inputs_adv_grad / (torch.norm(inputs_adv_grad, dim=-1, keepdim=True) +
                                                                 norm_epsilon)
        r = inputs_adv_tmp - inputs_adv
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        inputs_adv = inputs_adv + r
    else:  #SMART
        delta_grad_inputs = inputs_adv + inputs_adv_grad * alpha
        inputs_adv = delta_grad_inputs / (delta_grad_inputs.abs().max(-1, keepdim=True)[0] + norm_epsilon)
    return inputs_adv


class FGMPerturbation(object):

    def __init__(self, model, criterion, norm_type='l2', alpha=0.3, adv_alpha=None, **kwargs):
        self.model = model
        self.alpha = alpha
        self.criterion = criterion
        self.adv_alpha = adv_alpha
        self.norm_type = norm_type

    def forward(self, inputs, outputs, text_tensor, pred_size, length_tensor, loss):
        # TODO  为什么会报已经backward一次了的错，而PGD不会？？
        inputs_adv_grad, = torch.autograd.grad(loss, inputs, only_inputs=True, retain_graph=True)
        inputs_adv = norm_grad(inputs, inputs_adv_grad, self.norm_type, alpha=self.alpha)
        outputs_adv = self.model(inputs_adv)
        # 计算对抗样本的对抗损失
        outputs_adv = log_softmax(outputs_adv, dim=2)
        loss_adv = self.criterion(outputs_adv, text_tensor, pred_size, length_tensor)
        adv_alpha = 1 / max(1, loss_adv // loss) if self.adv_alpha is None else self.adv_alpha
        return loss_adv * adv_alpha + loss


class PGDPerturbation(object):

    def __init__(self,
                 model,
                 criterion,
                 norm_type='sign',
                 epsilon=0.3,
                 k=3,
                 alpha=0.01,
                 random_start=True,
                 adv_alpha=None,
                 **kwargs):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start
        self.criterion = criterion
        self.adv_alpha = adv_alpha
        self.norm_type = norm_type + '_pgd'

    def forward(self, inputs, outputs, text_tensor, pred_size, length_tensor, loss):
        if self.random_start:
            inputs_adv = torch.Tensor(np.random.uniform(-self.epsilon, self.epsilon, inputs.shape), device=inputs.device) + inputs
        else:
            inputs_adv = inputs
        for i in range(self.k):
            inputs_adv.requires_grad_()
            outputs_adv = self.model(inputs_adv)
            loss_adv = self.criterion(outputs_adv, text_tensor, pred_size, length_tensor)
            inputs_adv_grad, = torch.autograd.grad(loss_adv, inputs_adv, only_inputs=True, retain_graph=False)
            inputs_adv = norm_grad(inputs_adv,
                                   inputs_adv_grad,
                                   self.norm_type,
                                   self.epsilon,
                                   alpha=self.alpha,
                                   inputs_origin=inputs)
        outputs_adv = self.model(inputs_adv)
        # 计算对抗样本的对抗损失
        outputs_adv = log_softmax(outputs_adv, dim=2)
        loss_adv = self.criterion(outputs_adv, text_tensor, pred_size, length_tensor)
        adv_alpha = 1 / max(1, loss_adv // loss) if self.adv_alpha is None else self.adv_alpha
        return loss_adv * adv_alpha + loss


class FreeLBPerturbation(object):

    def __init__(self,
                 model,
                 criterion,
                 norm_type='sign',
                 epsilon=0.3,
                 k=3,
                 alpha=0.01,
                 random_start=True,
                 adv_alpha=None,
                 **kwargs):
        """
        Attack parameter initialization. The attack performs k steps of
        size alpha, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.random_start = random_start
        self.criterion = criterion
        self.adv_alpha = adv_alpha
        self.norm_type = norm_type + '_freelab'

    def forward(self, inputs, outputs, text_tensor, pred_size, length_tensor, loss):
        loss.backward(retain_graph=True)
        if self.random_start:
            inputs_adv = torch.Tensor(np.random.uniform(-self.epsilon, self.epsilon, inputs.shape), device=inputs.device) + inputs
        else:
            inputs_adv = inputs
        for i in range(self.k):
            inputs_adv.requires_grad_()
            outputs_adv = self.model(inputs_adv)
            loss_adv = self.criterion(outputs_adv, text_tensor, pred_size, length_tensor)
            adv_alpha = 1 / max(1, loss_adv // loss) if self.adv_alpha is None else self.adv_alpha
            loss_adv = loss_adv / self.k * adv_alpha
            inputs_adv_grad, = torch.autograd.grad(loss_adv, inputs_adv, only_inputs=False, retain_graph=False)
            inputs_adv = norm_grad(inputs_adv, inputs_adv_grad, self.norm_type, self.epsilon, alpha=self.alpha)
        outputs_adv = self.model(inputs_adv)
        # 计算对抗样本的对抗损失
        outputs_adv = log_softmax(outputs_adv, dim=2)
        loss_adv = self.criterion(outputs_adv, text_tensor, pred_size, length_tensor)
        adv_alpha = 1 / max(1, loss_adv // loss) if self.adv_alpha is None else self.adv_alpha
        return loss_adv / self.k * adv_alpha


class SmartPerturbation(object):
    """
    Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization
    criterion
    alpha noise扰动学习率
    epsilon 梯度scale时防止分母为0
    norm_type 梯度scale采用的范式
    noise_epsilon 扰动初始化系数
    https://github.com/namisan/mt-dnn/blob/ad6c3223f21955abfecdb722c452776a4b3052b5/train.py
    """

    def __init__(self,
                 model,
                 criterion,
                 epsilon=1e-6,
                 alpha=1e-5,
                 noise_epsilon=1e-5,
                 norm_type='inf',
                 k=1,
                 adv_alpha=None,
                 **kwargs):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.alpha = alpha
        self.K = k
        # sigma
        self.noise_epsilon = noise_epsilon
        self.norm_type = norm_type
        self.model = model
        self.criterion = criterion
        self.kldiv_loss = KLDivLoss(reduction='batchmean')
        self.adv_alpha = adv_alpha

    # 初始noise扰动 正态分布的扰动
    def generate_noise(self, inputs, noise_epsilon=1e-5):
        noise = inputs.data.new(inputs.size()).normal_(0, 1) * noise_epsilon
        noise.detach()
        noise.requires_grad_()
        return noise

    # 对称散度loss
    def stable_kl(self, logit, target, epsilon=1e-6, reduce=True):
        logit = logit.reshape(-1, logit.size(-1)).float()
        target = target.reshape(-1, target.size(-1)).float()
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()
        # return 0.5 * (self.kldiv_loss(logit, target) + self.kldiv_loss(target, logit))

    # 对抗loss输出
    def forward(self, inputs, outputs, text_tensor, pred_size, length_tensor, loss):
        # adv training
        # init delta
        # embed生成noise
        noise = self.generate_noise(inputs, noise_epsilon=self.noise_epsilon)
        inputs_adv = inputs + noise
        # noise更新K轮
        for step in range(0, self.K):
            # noise+embed得到对抗样本的输出adv_outputs
            outputs_adv = self.model(inputs_adv)
            loss_adv = self.stable_kl(outputs_adv, outputs.detach(), reduce=False)
            # 得到noise的梯度
            noise_grad, = torch.autograd.grad(loss_adv, noise, only_inputs=True, retain_graph=False)
            # 得到新的scale的noise
            noise = norm_grad(noise, noise_grad, self.norm_type, self.epsilon, self.alpha)
            noise = noise.detach()
            noise.requires_grad_()
            inputs_adv += noise
        outputs_adv = self.model(inputs_adv)
        # 计算对抗样本的对抗损失
        outputs_adv = log_softmax(outputs_adv, dim=2)
        loss_adv = self.criterion(outputs_adv, text_tensor, pred_size, length_tensor)
        adv_alpha = 1 / max(1, loss_adv // loss) if self.adv_alpha is None else self.adv_alpha
        return loss_adv * adv_alpha + loss
