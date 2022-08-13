import torch

from data import zidian_y, loader, zidian_xr, zidian_yr
from mask import mask_pad, mask_tril
from model import Transformer

# onnx-tf要求tensorflow==2.2.0
# onnx-tf==1.6
import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import argparse
import os

# 预测函数
def predict(x):
    print("预测输入:", x)
    # x = [1, 50]
    model.eval()

    # [1, 1, 50, 50]
    mask_pad_x = mask_pad(x)

    # 初始化输出,这个是固定值
    # [1, 50]
    # [[0,2,2,2...]]
    target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)

    # x编码,添加位置信息
    # [1, 50] -> [1, 50, 32]
    x = model.embed_x(x)

    # 编码层计算,维度不变
    # [1, 50, 32] -> [1, 50, 32]
    x = model.encoder(x, mask_pad_x)

    # 遍历生成第1个词到第49个词
    for i in range(49):
        # [1, 50]
        y = target

        # [1, 1, 50, 50]
        mask_tril_y = mask_tril(y)

        # y编码,添加位置信息
        # [1, 50] -> [1, 50, 32]
        y = model.embed_y(y)

        # 解码层计算,维度不变
        # [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,39分类
        # [1, 50, 32] -> [1, 50, 39]
        out = model.fc_out(y)

        # 取出当前词的输出
        # [1, 50, 39] -> [1, 39]
        out = out[:, i, :]

        # 取出分类结果
        # [1, 39] -> [1]
        # out = out.argmax(dim=1).detach()
        out = out.argmax(dim=1)

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target


model = Transformer()
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)

for epoch in range(1):
    for i, (x, y) in enumerate(loader):
        # x = [8, 50]
        # y = [8, 51]

        # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
        # [8, 50, 39]
        pred = model(x, y[:, :-1])

        # [8, 50, 39] -> [400, 39]
        pred = pred.reshape(-1, 39)

        # [8, 51] -> [400]
        y = y[:, 1:].reshape(-1)

        # 忽略pad
        select = y != zidian_y['<PAD>']
        pred = pred[select]
        y = y[select]

        loss = loss_func(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 200 == 0:
            # [select, 39] -> [select]
            pred = pred.argmax(1)
            correct = (pred == y).sum().item()
            accuracy = correct / len(pred)
            lr = optim.param_groups[0]['lr']
            print(epoch, i, lr, loss.item(), accuracy)

    print("----------------------")
    sched.step()
    # torch.save("/Users/zard/Documents/GitHub/Transformer_Example/my_test")

    # model = torch.nn.DataParallel(model)
    #
    # # 设置模型输入维度
    # input = torch.randn([1, 2, 3])
    #
    # # 设置输入张量名，多个输入就是多个名
    # input_names = ["input"]
    #
    # # 设置输出张量名
    # output_names = ["output"]
    #
    # # 自定义onnx文件名和路径
    # # onnx_filename = "model.onnx"
    # onnx_filename = "/Users/zard/Documents/GitHub/Transformer_Example/my_test/model.onnx"
    #
    # # 执行转化和保存
    # torch.onnx.export(model.module, input, onnx_filename, verbose=True, input_names=input_names,
    #                   output_names=output_names)
    #
    # #
    # onnx_model = onnx.load("model.onnx")  # load onnx model
    # tf_exp = prepare(onnx_model)  # prepare tf representation
    # tf_exp.export_graph("model.pb")  # export the model
    print("----------------------")

# 测试
for i, (x, y) in enumerate(loader):
    break

for i in range(8):
    print(i)
    print("输入参数:", x[i])
    print(''.join([zidian_xr[i] for i in x[i].tolist()]))
    print(''.join([zidian_yr[i] for i in y[i].tolist()]))
    print(''.join([zidian_yr[i] for i in predict(x[i].unsqueeze(0))[0].tolist()]))
    print('-------------------')


