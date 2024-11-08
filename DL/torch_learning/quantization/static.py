#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   static.py
@Time   :   2024/11/05 12:05:23
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   通过PTQ static量化MobileNetV2走一遍torch量化流程
            参考链接：https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from datasets import load_dataset


# ================== 1. 定义模型结构 =======================
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """fused module
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        """Difference between ReLU and ReLU6
        ReLU:  y = max(0, x)
        ReLU6: y = min(max(0, x), 6)
        ReLU6限定输出在[0, 6]之间，量化后的模型中用的多，限定范围能够提供更好的数值稳定性
        """
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding,
                groups=groups, bias=False
            ),
            nn.BatchNorm2d(
                out_planes, momentum=0.1
            ),
            # Replace with ReLU
            nn.ReLU(
                inplace=False
            )
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio) -> None:
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        # TODO 为什么是stride = 1
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw: pointwise convolution, 即1x1卷积
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1)
            )
        layers.extend(
            [
                # dw: depthwise convolution, 即3x3卷积
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-Leaner
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup, momentum=0.1)
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)
        
    
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8
    ):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # Build first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # Build last serval layers
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        )
        self.features = nn.Sequential(*features)
        # Insert quantstub and dequantstub
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # Build classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """static需要在整个网络输入后首先quant，输出之前dequant
        """
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])  # output shape (batch_size, num_channel)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self, is_qat=False):
        """
        Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
        This operation does not change the numerics
        """
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        for m in self.modules():  # all modules inculdes nested modules
            if type(m) == ConvBNReLU:
                # 之前的类只是定义，实际forward还是for循环模块依次forward
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:  # ConvBNReLU会进入上一个if的
                        fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


# ================== 2. 定义helper ========================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for datadict in data_loader:
            image, target = datadict.values()
        # for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5


def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file, weights_only=True)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


# ======================== 3. data ============================
def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(
        # for each channel
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    """ImageNet方式sampler之后返回的是list，就是load_dataset方法的list(valus())
    """
    # dataset = torchvision.datasets.ImageNet(
    #     root=data_path, split="train",
    #     transform=transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    # )
    # dataset_test = torchvision.datasets.ImageNet(
    #     data_path, split="val",
    #     transform=transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )

    def transform_train(examples):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        new_examples = {"image": [], "label": []}
        for img, label in zip(examples["image"], examples["label"]):
            if len(img.split()) != 3:
                continue
            new_examples["image"].append(transform(img))
            new_examples["label"].append(label)
        return new_examples
            
    def transform_valid(examples):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        new_examples = {"image": [], "label": []}
        for img, label in zip(examples["image"], examples["label"]):
            if len(img.split()) != 3:
                continue
            new_examples["image"].append(transform(img))
            new_examples["label"].append(label)
        return new_examples

    dataset = load_dataset(data_path, split="train")
    dataset.set_transform(transform_train)
    dataset_test = load_dataset(data_path, split="valid")
    dataset_test.set_transform(transform_valid)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler
    )

    return data_loader, data_loader_test


# ======================== 4. float baseline ==================
def report_baseline():
    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    float_model.eval()

    # Fuses modules
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)


# ======================== 5. PTQ =============================
def ptq():
    """
    Specify quantization configuration
    Start with simple min/max range estimation and per-tensor quantization of weights
    """
    # 1）算子融合
    float_model.eval()
    float_model.fuse_model()

    # 2）选择observer
    # MinMax
    float_model.qconfig = torch.ao.quantization.default_qconfig
    # Histogram + PerChannelMinMax
    # float_model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    print(float_model.qconfig)

    # 3）插入observer
    torch.ao.quantization.prepare(float_model, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', float_model.features[1].conv)

    # 4）前向observer统计量化参数
    # Calibrate with the training set
    evaluate(float_model, criterion, data_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # 5）量化模型
    # Convert to quantized model
    torch.ao.quantization.convert(float_model, inplace=True)
    # You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
    # This warning occurs because not all modules are run in each model runs, so some
    # modules may not be calibrated.
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',float_model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))    


if __name__ == "__main__":
    data_path = '/media/data/datasets/TinyImageNet'
    saved_model_dir = '/media/data/Quantization/PTQ/'
    float_model_file = 'mobilenet_pretrained_float.pth'
    scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
    scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

    train_batch_size = 320
    eval_batch_size = 200
    num_eval_batches = 10000
    num_calibration_batches = 640

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')

    # report_baseline()
    ptq()