import os
import torch
from torch import nn
import torch.nn.functional as F
import gfl.core.strategy as strategy
from gfl.core.job_manager import JobManager
from torchvision.datasets import CIFAR10


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet56(nn.Module):

    def __init__(self):
        self.block = Bottleneck
        self.layers = [6, 6, 6]
        self.num_classes = 10
        self.zero_init_residual = False
        self.groups = 1
        self.width_per_group = 64
        self.replace_stride_with_dilation = None
        self.norm_layer = None
        self.KD = False
        super(ResNet56, self).__init__()
        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        self._norm_layer = self.norm_layer

        self.inplanes = 16
        self.dilation = 1
        if self.replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            self.replace_stride_with_dilation = [False, False, False]
        if len(self.replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(self.replace_stride_with_dilation))

        # self.groups = groups
        self.base_width = self.width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(self.block, 16, self.layers[0])
        self.layer2 = self._make_layer(self.block, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 64, self.layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * self.block.expansion, self.num_classes)
        # self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = self.fc(x_f)  # B x num_classes
        if self.KD == True:
            return x_f, x
        else:
            return x


# def resnet56(class_num, pretrained=False, path=None, **kwargs):
#     """
#     Constructs a ResNet-110 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained.
#     """
#
#     if pretrained:
#         checkpoint = torch.load(path)
#         state_dict = checkpoint['state_dict']
#
#         from collections import OrderedDict
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             # name = k[7:]  # remove 'module.' of dataparallel
#             name = k.replace("module.", "")
#             new_state_dict[name] = v
#
#         model.load_state_dict(new_state_dict)
#     return model


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    # model = model.to(device)

    job_manager = JobManager()
    job = job_manager.generate_job(work_mode=strategy.WorkModeStrategy.WORKMODE_STANDALONE,
                                   fed_strategy=strategy.FederateStrategy.FED_DISTILLATION, epoch=100, distillation_alpha=0.5, model=CNN)
    job_manager.submit_job(job, model)

    # train_dataset = torch.load(os.path.join("./data", "train_dataset_0"))
    # test_dataset = torch.load(os.path.join("./data", "test_dataset"))
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,
    #                                                batch_size=64,
    #                                                shuffle=True,
    #                                                num_workers=0,
    #                                                pin_memory=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset,
    #                                                batch_size=64,
    #                                                shuffle=True,
    #                                                num_workers=0,
    #                                                pin_memory=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-3)
    # acc = 0
    # # print(model)
    # for _ in range(100):
    #     model.train()
    #     for idx, (batch_data, batch_target) in enumerate(train_dataloader):
    #         batch_data, batch_target = batch_data.to(device), batch_target.to(device)
    #         # print(batch_data.shape)
    #         pred = model(batch_data)
    #         # log_pred = torch.log(F.softmax(pred, dim=1))
    #         loss = F.cross_entropy(pred, batch_target.long())
    #         batch_acc = torch.eq(pred.argmax(dim=1), batch_target).sum().float().item()
    #         acc += batch_acc
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # if idx % 200 == 0:
    #             # accuracy_function = train_model.get_train_strategy().get_accuracy_function()
    #             # if accuracy_function is not None and isfunction(accuracy_function):
    #             #     accuracy = accuracy_function(pred, batch_target)
    #             # self.logger.info("train_loss: {}, train_acc: {}".format(loss.item(), float(batch_acc) / len(batch_target)))
    #     model.eval()
    #     acc = 0
    #     for idx, (batch_data, batch_target) in enumerate(test_dataloader):
    #         batch_data, batch_target = batch_data.to(device), batch_target.to(device)
    #         # print(batch_data.shape)
    #         pred = model(batch_data)
    #         # log_pred = torch.log(F.softmax(pred, dim=1))
    #         test_loss = F.cross_entropy(pred, batch_target.long())
    #         batch_acc = torch.eq(pred.argmax(dim=1), batch_target).sum().float().item()
    #         acc += batch_acc
    #         optimizer.zero_grad()
    #         test_loss.backward()
    #         optimizer.step()
    #
    #     print("test_loss: {}, test_acc: {}".format(test_loss.item(), float(acc)/len(test_dataset)))