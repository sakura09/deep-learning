import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 预测一个锚框的类别
# nums_imput输入通道数， num_anchors锚框数，num_class类别数
# 之前的预测用的是dense全连接层，但是这里用的是卷积分类，在NiN中有过，NiN中的输出通道数就是类别，高宽经过pooling后变成1，这里的kernel和padding没有改变输出高宽，对每一个像素做预测
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),  #输出通道数 = 锚框数 *（类别数 + 1背景）
                     kernel_size=3, padding=1)

# 边界框预测层， 对与真实边界框的offset做预测，offset是4个值，分别是对于锚框中心x,y的偏移，高宽的偏移。 所以这里的输出通道数 = anchors * 4
# 
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 连接多尺度的预测
def forward(x, block):
    
    return block(x)
# 参数是batchzise，通道数，fmap的高宽
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
# 把4D的变成2D， 
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)  #batchsize保持，高宽前移，通道数挪到最后，startI_dim=1意思把后面的3维拉成一个向量

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)  #在宽上concat

# 定义网络
# 可以用pre_train的，这里方便跑重新定义一个很小的cnn, 作用就是每次高宽减半
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# 基本网络块，从原始图片抽特征到第一次做锚框
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]       #输入channel数，第一次3，增长到16，到32，到64 
    for i in range(len(num_filters) - 1):     #做了3次blk，高宽减少8倍
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape

# 整个单发多框检测由5个模块组成
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

# 对每个块定义前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)    #这个stage下的fmap
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# 超参数 5个stage
sizes = [[0.2, 0.272], 
         [0.37, 0.447], 
         [0.54, 0.619], 
         [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1  

# 定义完整模型
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)  #reshape一下，成3d，增加一维对每个类的预测值方便后面做softmax
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
    
# 创建一个模型实例，看看长啥样
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)

# 训练模型
# 读取香蕉集
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
# 模型初始化和优化算法
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评估函数
cls_loss = nn.CrossEntropyLoss(reduction='none') #reduction=nono不要把这些东西加起来    
bbox_loss = nn.L1Loss(reduction='none') #L1loss当你预测不靠谱的时候也不会给一个特别大的损失

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),    #这里是对每个锚框做loss，reshape(-1)把前面的批量大小和锚框个数放到一起；label同理
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,      #mask的意思，当锚框是背景框的时候mask是0，不然是1
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        #这个Y是真实物体的bbox，不能预测
        X, Y = features.to(device), target.to(device)   
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)  #这三个是预测值，与真实的Y做对比   
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')


# 预测目标
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

# 我们筛选所有置信度不低于0.9的边界框，做为最终输出
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)