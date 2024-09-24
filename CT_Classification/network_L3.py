import torch.nn as nn
import torchvision.models as models
import torch

"""
    目的：搭建多骨干网络结构
    输入：is_pretrained：is_pretrained=True迁移预训练权重, is_pretrained=False不加载预训练权重
"""

# 初始化模型
def initialize_model(backbone, pretrained, NUM_CLASS=2):
    if backbone == "alexnet":
        model_conv = models.alexnet(pretrained=pretrained)
        num_ftrs = model_conv.classifier[6].in_features
        model_conv.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == 'resnet18':
        model_conv = models.resnet18(pretrained=pretrained)
        # for param in model_conv.parameters():
        #     param.requires_grad = False  # 冻结所有层，只训练fc层
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, NUM_CLASS)  # 修改fc层

    elif backbone == 'resnet34':
        model_conv = models.resnet34(pretrained=pretrained)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == 'resnet50':
        model_conv = models.resnet50(pretrained=pretrained)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == 'resnet101':
        model_conv = models.resnet101(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == "vgg11":
        model_conv = models.vgg11(pretrained=pretrained)
        num_ftrs = model_conv.classifier[6].in_features
        model_conv.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == 'vgg16':
        model_conv = models.vgg16(pretrained=pretrained)
        num_ftrs = model_conv.classifier[6].in_features
        model_conv.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == "vgg19":
        model_conv = models.vgg19(pretrained=pretrained)
        num_ftrs = model_conv.classifier[6].in_features
        model_conv.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)


    elif backbone == "densenet":
        model_conv = models.densenet121(pretrained=pretrained)
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_conv = models.inception_v3(pretrained=pretrained)
        # Handle the auxilary net
        num_ftrs = model_conv.AuxLogits.fc.in_features
        model_conv.AuxLogits.fc = nn.Linear(num_ftrs, NUM_CLASS)
        # Handle the primary net
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, NUM_CLASS)
        model_conv.aux_logits = False

    model = model_conv.cuda() if torch.cuda.is_available() else model_conv
    return model
