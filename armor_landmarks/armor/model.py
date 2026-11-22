import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ArmorNet(nn.Module):
    def __init__(self, num_classes=6, max_points=12, pretrained=False):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layers = nn.ModuleList([backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4])
        self.lateral = nn.ModuleList([nn.Conv2d(c,64,1) for c in [64,128,256,512]])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU()) for _ in range(3)])
        self.classifier = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,num_classes+1))
        self.landmarks = nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,num_classes*max_points,1))
        self.visibility = nn.Linear(512,num_classes*max_points)
        self.num_classes, self.max_points = num_classes, max_points

    def forward(self, x):
        x = self.stem(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        pooled = features[-1].mean(dim=(-2,-1))
        fused = self.lateral[-1](features[-1])
        for i in [2,1,0]:
            fused = F.interpolate(fused,size=features[i].shape[-2:],mode='bilinear',align_corners=False)
            fused = self.smooth[i](fused+self.lateral[i](features[i]))
        heat = self.landmarks(fused)
        heat = heat.reshape(x.shape[0],self.num_classes,self.max_points,*heat.shape[-2:])
        return self.classifier(pooled), heat, self.visibility(pooled).reshape(-1,self.num_classes,self.max_points)


def losses(outputs, target):
    classes, heatmaps, visibility = outputs
    labels = target['class_id']
    cls_loss = F.cross_entropy(classes,labels)
    positive = labels < heatmaps.shape[1]
    if not positive.any():
        zero = heatmaps.sum()*0 + visibility.sum()*0
        return cls_loss+zero, {'cls':cls_loss.detach(),'heat':zero.detach(),'vis':zero.detach()}
    batch = torch.arange(len(labels),device=labels.device)[positive]
    selected = heatmaps[batch,labels[positive]].flatten(2)
    wanted = target['heatmaps'][positive].flatten(2)
    mask = target['visible'][positive]*target['exists'][positive]
    heat_loss = (-(wanted*F.log_softmax(selected,dim=-1)).sum(-1)*mask).sum()/mask.sum().clamp_min(1)
    vis_loss = F.binary_cross_entropy_with_logits(visibility[batch,labels[positive]],target['visible'][positive],reduction='none')
    exists = target['exists'][positive]
    vis_loss = (vis_loss*exists).sum()/exists.sum().clamp_min(1)
    total = cls_loss+heat_loss+0.25*vis_loss
    return total, {'cls':cls_loss.detach(),'heat':heat_loss.detach(),'vis':vis_loss.detach()}
