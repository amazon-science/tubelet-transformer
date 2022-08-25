'''
CSN-50
The code refers to https://github.com/dmlc/gluon-cv
Modified by Zhang Yanyi
'''


import torch
import torch.nn as nn


eps = 1e-3
bn_mmt = 0.1


class Affine(nn.Module):
    def __init__(self, feature_in):
        super(Affine, self).__init__()
        self.weight = nn.Parameter(torch.randn(feature_in, 1, 1, 1))
        self.bias = nn.Parameter(torch.randn(feature_in,1, 1, 1))
        self.weight.requires_grad = False
        self.bias.requires_grad = False


    def forward(self, x):
        x = x * self.weight + self.bias
        return x


class ResNeXtBottleneck(nn.Module):
    # expansion = 2

    def __init__(self, in_planes, planes, stride=1, temporal_stride=1,
                 down_sample=None, expansion=2, temporal_kernel=3, use_affine=True):

        super(ResNeXtBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 1, 1), bias=False, stride=(1, 1, 1))

        if use_affine:
            self.bn1 = Affine(planes)
        else:
            self.bn1 = nn.BatchNorm3d(planes, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), bias=False,
                               stride=(temporal_stride, stride, stride),
                               padding=((temporal_kernel - 1) // 2, 1, 1),
                               groups=planes)

        if use_affine:
            self.bn3 = Affine(planes)
        else:
            self.bn3 = nn.BatchNorm3d(planes, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.conv4 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)

        if use_affine:
            self.bn4 = Affine(planes * self.expansion)
        else:
            self.bn4 = nn.BatchNorm3d(planes * self.expansion, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self,
                 block,
                 sample_size,
                 sample_duration,
                 block_nums,
                 num_classes=400,
                 use_affine=True,
                 last_stride=True):

        self.use_affine = use_affine
        self.in_planes = 64
        self.num_classes = num_classes

        super(ResNeXt, self).__init__()

        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False)
        if use_affine:
            self.bn1 = Affine(64)
        else:
            self.bn1 = nn.BatchNorm3d(64, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, in_planes=64, planes=64, blocks=block_nums[0],
                                       stride=1, expansion=4)

        self.layer2 = self._make_layer(block, in_planes=256, planes=128, blocks=block_nums[1],
                                       stride=2, temporal_stride=2, expansion=4)

        self.layer3 = self._make_layer(block, in_planes=512, planes=256, blocks=block_nums[2],
                                       stride=2, temporal_stride=2, expansion=4)

        last_stride = 2 if last_stride else 1
        print("last stride: {}".format(last_stride))
        self.layer4 = self._make_layer(block, in_planes=1024, planes=512, blocks=block_nums[3],
                                       stride=last_stride, temporal_stride=2, expansion=4)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.out_fc = nn.Linear(in_features=2048, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self,
                    block,
                    in_planes,
                    planes,
                    blocks,
                    stride=1,
                    temporal_stride=1,
                    expansion=4):

        if self.use_affine:
            down_bn = Affine(planes * expansion)
        else:
            down_bn = nn.BatchNorm3d(planes * expansion, track_running_stats=True, eps=eps, momentum=bn_mmt)
        down_sample = nn.Sequential(
            nn.Conv3d(
                in_planes,
                planes * expansion,
                kernel_size=1,
                stride=(temporal_stride, stride, stride),
                bias=False), down_bn)
        layers = []
        layers.append(
            block(in_planes, planes, stride, temporal_stride, down_sample, expansion,
                  temporal_kernel=3, use_affine=self.use_affine))
        for i in range(1, blocks):
            layers.append(block(planes * expansion, planes, expansion=expansion,
                                temporal_kernel=3, use_affine=self.use_affine))

        return nn.Sequential(*layers)

    def forward(self, x):

        bs, _, _, _, _ = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(bs, -1)
        # logits = self.sigmoid(self.out_fc(x))

        return x, None


def build_model(n_classes=400,
                sample_size=224,
                sample_duration=128,
                pretrain_path="",
                load_fc=True,
                load_pretrain=True,
                use_affine=True,
                tune_point=5,
                last_stride=True):

    model = ResNeXt(ResNeXtBottleneck,
                    num_classes=n_classes,
                    sample_size=sample_size,
                    block_nums=[3, 4, 6, 3],
                    sample_duration=sample_duration,
                    use_affine=use_affine,
                    last_stride=last_stride)

    if load_pretrain and n_classes == 400 and load_fc:
        load_weights(model, pretrain_path=pretrain_path, load_fc=load_fc, use_affine=use_affine, tune_point=tune_point)
    elif load_pretrain:
        load_weights(model, pretrain_path=pretrain_path, load_fc=False, use_affine=use_affine, tune_point=tune_point)

    return model


def copy_weights(layer_weight, weights_name, weights, weights_dict_copy):
    assert layer_weight.shape == torch.from_numpy(weights).shape
    layer_weight.data.copy_(torch.from_numpy(weights))
    weights_dict_copy.pop(weights_name)


def copy_bn(layer, layer_name, weights, weights_dict_copy, use_affine=True):
    if use_affine:
        copy_weights(layer_weight=layer.weight, weights_name=layer_name + "_s",
                     weights=weights[layer_name + "_s"].reshape(-1, 1, 1, 1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.bias, weights_name=layer_name + "_b",
                     weights=weights[layer_name + "_b"].reshape(-1, 1, 1, 1),
                     weights_dict_copy=weights_dict_copy)
    else:
        copy_weights(layer_weight=layer.weight, weights_name=layer_name + "_s",
                     weights=weights[layer_name + "_s"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.bias, weights_name=layer_name + "_b",
                     weights=weights[layer_name + "_b"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.running_mean, weights_name=layer_name + "_rm",
                     weights=weights[layer_name + "_rm"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)
        copy_weights(layer_weight=layer.running_var, weights_name=layer_name + "_riv",
                     weights=weights[layer_name + "_riv"].reshape(-1),
                     weights_dict_copy=weights_dict_copy)


def load_weights(model, pretrain_path, load_fc=True, use_affine=False, tune_point=5):
    import scipy.io as sio
    print("load weights plus")
    r_weights = sio.loadmat(pretrain_path)
    r_weights_copy = sio.loadmat(pretrain_path)

    conv1 = model.conv1
    conv1_bn = model.bn1

    if tune_point > 1:
        conv1.weight.requires_grad = False
        for param in conv1_bn.parameters():
            param.requires_grad = False

    res2 = model.layer1
    res3 = model.layer2
    res4 = model.layer3
    res5 = model.layer4

    stages = [res2, res3, res4, res5]

    copy_weights(layer_weight=conv1.weight, weights_name='conv1_w', weights=r_weights['conv1_w'],
                 weights_dict_copy=r_weights_copy)
    copy_bn(layer=conv1_bn, layer_name="conv1_spatbn_relu",
            weights=r_weights, weights_dict_copy=r_weights_copy, use_affine=use_affine)

    start_count = [0, 3, 7, 13]

    for s in range(len(stages)):
        res = stages[s]._modules
        count = start_count[s]
        # print(count)
        for k, block in res.items():
            # load
            copy_weights(layer_weight=block.conv1.weight, weights_name="comp_{}_conv_1_w".format(count),
                         weights=r_weights["comp_{}_conv_1_w".format(count)], weights_dict_copy=r_weights_copy)
            copy_weights(layer_weight=block.conv3.weight, weights_name="comp_{}_conv_3_w".format(count),
                         weights=r_weights["comp_{}_conv_3_w".format(count)], weights_dict_copy=r_weights_copy)
            copy_weights(layer_weight=block.conv4.weight, weights_name="comp_{}_conv_4_w".format(count),
                         weights=r_weights["comp_{}_conv_4_w".format(count)], weights_dict_copy=r_weights_copy)

            copy_bn(layer=block.bn1, layer_name="comp_{}_spatbn_1".format(count),
                    weights=r_weights, weights_dict_copy=r_weights_copy, use_affine=use_affine)
            copy_bn(layer=block.bn3, layer_name="comp_{}_spatbn_3".format(count), weights=r_weights,
                    weights_dict_copy=r_weights_copy, use_affine=use_affine)
            copy_bn(layer=block.bn4, layer_name="comp_{}_spatbn_4".format(count), weights=r_weights,
                    weights_dict_copy=r_weights_copy, use_affine=use_affine)

            if block.down_sample is not None:
                down_conv = block.down_sample._modules["0"]
                down_bn = block.down_sample._modules["1"]

                copy_weights(layer_weight=down_conv.weight, weights_name="shortcut_projection_{}_w".format(count),
                             weights=r_weights["shortcut_projection_{}_w".format(count)], weights_dict_copy=r_weights_copy)
                copy_bn(layer=down_bn, layer_name="shortcut_projection_{}_spatbn".format(count),
                        weights=r_weights, weights_dict_copy=r_weights_copy, use_affine=use_affine)

            count += 1
        if tune_point > s + 2:
            for param in stages[s].parameters():
                param.requires_grad = False

    if load_fc:
        copy_weights(layer_weight=model.out_fc.weight,
                     weights_name='last_out_L400_w', weights=r_weights['last_out_L400_w'],
                     weights_dict_copy=r_weights_copy)
        copy_weights(layer_weight=model.out_fc.bias,
                     weights_name='last_out_L400_b',
                     weights=r_weights['last_out_L400_b'][0],
                     weights_dict_copy=r_weights_copy)

    print("load pretrain model " + pretrain_path)
    print("load fc", load_fc)
    for k, v in r_weights_copy.items():
        if "momentum" not in k and "model_iter" not in k and "__globals__" not in k and "__header__" not in k and "lr" not in k and "__version__" not in k:
            print(k, v.shape)


def build_CSN(cfg):
    tune_point = 4
    model = build_model(n_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                        sample_size=cfg.CONFIG.DATA.IMG_SIZE,
                        sample_duration=cfg.CONFIG.MODEL.TEMP_LEN,
                        pretrain_path=cfg.CONFIG.MODEL.PRETRAIN_BACKBONE_DIR,
                        load_fc=False,
                        load_pretrain=cfg.CONFIG.MODEL.PRETRAINED,
                        use_affine=False,
                        tune_point=tune_point,
                        last_stride=cfg.CONFIG.MODEL.LAST_STRIDE)
    print("tune point: {}".format(tune_point))
    return model

