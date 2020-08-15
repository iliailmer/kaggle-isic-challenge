import kornia
from torch.nn import functional as F
import torch
import numpy as np
from torch import Tensor
from torch import nn
from efficientnet_pytorch.utils import MemoryEfficientSwish as Swish
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    """ResNet-Based Model."""

    def __init__(self, resnet_model, num_classes=2):
        """Initialize."""
        super().__init__()
        self.features = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
            resnet_model.layer2,
            resnet_model.layer3,
            resnet_model.layer4,
            resnet_model.avgpool
        )

        self.classifier = nn.Linear(2048+3, num_classes)
        # nn.Sequential(nn.Linear(2048+3, 128),
        #                                 nn.PReLU(),
        #                                 nn.Linear(128, 2))

        # self.meta_clf = nn.Sequential(nn.Linear(3, 16),
        #                               Swish(),
        #                               nn.Linear(16, 8),
        #                               nn.PReLU(),
        #                               nn.Linear(8, 3),
        #                               Swish())
        nn.init.xavier_uniform_(self.classifier.weight)
        # self.init(self.meta_clf)

        # nn.init.xavier_uniform_(self.classifier[2].weight)

    def init(self, layer: nn.Sequential):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, features, meta_features):
        """Run Forward pass."""
        features = self.features(features)
        features = features.view(features.shape[0], -1)
        # meta_features = self.meta_clf(meta_features)
        features = torch.cat([features, meta_features], dim=1)
        features = self.classifier(features)

        assert len(features[0].shape) >= 1, f"{features.shape}"
        return features


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class ENet(nn.Module):
    def __init__(self, name: str, num_classes=2):
        super().__init__()
        self.model = EfficientNet.from_pretrained(name)
        self.dense_output = nn.Linear(
            self.model._fc.in_features+3, num_classes)
        self.meta_clf = nn.Sequential(nn.Linear(3, 16),
                                      Swish(),
                                      nn.Linear(16, 8),
                                      nn.PReLU(),
                                      nn.Linear(8, 3),
                                      Swish())
        self.init(self.meta_clf)

    def forward(self, features, meta_features):
        features = self.model.extract_features(features)
        features = F.avg_pool2d(
            features, features.size()[2:]).reshape(features.shape[0], self.model._fc.in_features)
        meta_features = self.meta_clf(meta_features)
        features = torch.cat([features, meta_features], dim=1)
        features = self.dense_output(features)
        return features

    def init(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)


class HarmonicBlock(nn.Module):
    def __init__(self, input_channels, output_ch,
                 kernel_size=3,
                 pad=1,
                 stride=1,
                 t=2,
                 alpha_root=1.0,
                 lmbda=None,
                 diag=False,
                 bn=True,
                 dropout=False,
                 bias=False,
                 use_res=True):
        super(HarmonicBlock, self).__init__()
        """
        :param input_channels: number of channels in the input
        :param kernel_size: size of the kernel in the filter bank
        :param pad: padding size
        :param stride: stride size
        :param lmbda: number of filters to be actually used
        """
        self.bn = bn
        self.drop = dropout
        self.input_channels = input_channels
        self.output_ch = output_ch
        self.stride = stride
        self.K = kernel_size
        if kernel_size % 2 == 0:
            self.pad = kernel_size // 2
        else:
            self.pad = pad
        self.diag = diag
        # preferably to have N=K !!
        # (to fully replicate the paper), this is the convolution window size
        self.N = self.K
        self.PI = torch.as_tensor(np.pi)
        self.use_res = use_res
        self.alpha_root = alpha_root
        if lmbda is not None:
            if lmbda > self.K ** 2:
                self.lmbda = self.K ** 2  # limits the number of kernels
            else:
                self.lmbda = lmbda
        else:
            self.lmbda = lmbda
        self.diag = diag  # flag to select diagonal entries of the block
        self.filter_bank = self.get_filter_bank(N=self.N,
                                                K=self.K,
                                                # kernel size
                                                input_channels=self.input_channels,
                                                t=t,  # type of DCT
                                                lmbda=self.lmbda,
                                                diag=self.diag).float().cuda()
        self.conv = nn.Conv2d(in_channels=self.filter_bank.shape[0],
                              out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              bias=bias)
        if (stride != 2 or input_channels != output_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels,
                          output_ch,
                          kernel_size=2 if self.K % 2 == 0 else 1,
                          stride=stride,
                          padding=1 if self.K % 2 == 0 else 0,
                          bias=False),
                nn.BatchNorm2d(output_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(0.5)

    @staticmethod
    def dct_matrix(t=1, N=32):
        if t == 1:
            # N- the size of the input
            # n is the column dummy index, k is the row dummy index
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for p in range(N):
                res[p, -1] = (-1) ** p
            for n in range(1, N - 1):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N - 1) * n * k)
            return res
        if t == 2:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * k)
            return res
        if t == 3:
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for n in range(1, N):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N) * n * (k + 0.5))
            return res
        if t == 4:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * (k + 0.5))
            return res

    def filter_from_dct_matrix(self, i, j, size, t=2):
        mat = self.dct_matrix(N=size, t=t)
        fltr = mat[i, :].reshape((-1, 1)).dot(mat[j, :].reshape(1, -1))
        return torch.as_tensor(fltr)

    def fltr(self, u, v, N, k):
        return torch.as_tensor([[torch.cos(torch.as_tensor(self.PI / N * (ii + 0.5) * v)) * torch.cos(
            torch.as_tensor(self.PI / N * (jj + 0.5) * u)) for ii in range(k)] for jj in range(k)])

    def get_idx(self, K, l):
        out = []
        for i in range(K):
            for j in range(K):
                if i + j < l:
                    out.append(K * i + j)
        return tuple(out)

    def get_idx_diag(self, K):
        out = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    out.append(i + j)
        return tuple(out)

    def draw_filters(self, fb_=None):
        if fb_ is None:
            fb_ = self.filter_bank
        fig, ax = plt.subplots(len(fb_), 1, figsize=(12, 4))
        j = 0
        for i in range(len(fb_)):
            ax[i].imshow(fb_[i, 0, :, :])
            ax[i].axis('off')
            ax[i].grid(False)

    def get_filter_bank(self, N, K, input_channels=3, t=2, lmbda=None, diag=False):
        filter_bank = torch.zeros((K, K, K, K))
        for i in range(K):
            for j in range(K):
                filter_bank[i, j, :, :] = self.filter_from_dct_matrix(
                    i, j, K, t)  # self.fltr(i, j, N, K)
        if lmbda is not None:
            ids = self.get_idx(K, lmbda)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        if diag:
            ids = self.get_idx_diag(K)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        return torch.stack(tuple([filter_bank.view(-1, 1, K, K)] * input_channels), dim=0).view(
            (-1, 1, K, K))

    def alpha_rooting(self, x, alpha=1.0):
        if alpha is not None:
            return x.sign() * torch.abs(x).pow(alpha)
        else:
            return x

    def forward(self, x):
        in_ = x
        x = F.conv2d(x.float(),
                     weight=self.filter_bank,
                     padding=self.pad,
                     stride=self.stride,
                     groups=self.input_channels)  # int(self.K/2)
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
            # x = self.alpha_rooting(x, alpha=self.alpha_root)
            # + self.shortcut(in_)
            # x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
            # x = self.alpha_rooting(x, alpha=self.alpha_root)
        x = F.relu(x)
        return x


class WideOrthoResNet(nn.Module):
    def __init__(self, in_channels, kernel_size=None,
                 depth=10, num_classes=10, widen_factor=1,
                 block=HarmonicBlock, bn=True, drop=False, droprate=0.1,
                 alpha_root=None,
                 lmbda=None, diag=False):
        super(WideOrthoResNet, self).__init__()
        nChannels = [16, 16 * widen_factor,
                     32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        self.lmbda = lmbda
        self.alpha_root = alpha_root
        self.diag = diag
        self.bn = bn
        self.dropout = drop  # extra dropout inside the block
        pad = kernel_size // 2
        self.conv1 = block(input_channels=in_channels, bn=self.bn,
                           dropout=self.dropout,
                           output_ch=nChannels[0],
                           kernel_size=kernel_size,
                           alpha_root=alpha_root,
                           stride=1,
                           pad=pad, bias=False)
        self.drop = nn.Dropout(droprate)  # ORIGINAL=0.1

        self.stack1 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[0],
                                       out_planes=nChannels[1],
                                       kernel_size=kernel_size,  # //2,
                                       stride=1,
                                       pad=pad)
        self.stack2 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[1],
                                       out_planes=nChannels[2],
                                       kernel_size=kernel_size,
                                       stride=2,
                                       pad=pad)
        self.stack3 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[2],
                                       out_planes=nChannels[3],
                                       kernel_size=kernel_size,
                                       stride=2,
                                       pad=pad)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        #self.fc_64 = nn.Linear(nChannels[3] * 4, num_classes)
        self.nChannels = nChannels[3]

        # self.center = DecoderBlock(nChannels[3], nChannels[3], nChannels[3])
        # self.dec1 = DecoderBlock(nChannels[3]+nChannels[2], nChannels[1], 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, in_planes, out_planes,
                    nb_layers, stride, kernel_size, pad):
        strides = [stride] + [1] * (nb_layers - 1)
        stacking = []
        for st in strides:
            stacking.append(block(input_channels=in_planes,
                                  output_ch=out_planes,
                                  lmbda=self.lmbda,
                                  diag=self.diag,
                                  bn=self.bn,
                                  dropout=self.dropout,
                                  # extra dropout inside the block
                                  kernel_size=kernel_size,
                                  alpha_root=self.alpha_root,
                                  stride=st,
                                  pad=pad))
            # stacking.append(nn.Dropout(0.5))
            if in_planes != out_planes:
                in_planes = out_planes
        return nn.Sequential(*stacking)

    # def _num_parameters(self, trainable=True):
    #     k = 0
    #     all_ = 0
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
    #             if m.weight.requires_grad:
    #                 all_ += m.weight.size().numel()
    #         if isinstance(m, HarmonicBlock) or isinstance(m, HadamardBlock) or isinstance(m, SlantBlock):
    #             all_ += m.filter_bank.size().numel()
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
    #             if m.weight.requires_grad:
    #                 k += m.weight.size().numel()
    #     return k, all_

    def forward(self, x):
        conv1 = self.drop(self.conv1(x))
        stack1 = self.drop(self.stack1(conv1))
        stack2 = self.drop(self.stack2(stack1))
        stack3 = self.drop(self.stack3(stack2))
        bn = self.relu(self.bn1(stack3))
        out = F.avg_pool2d(bn, bn.shape[-1])
        out = self.fc(out.view(-1, self.nChannels))
        return out
