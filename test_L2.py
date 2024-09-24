import math

import torch.optim
from pylab import *
import dataset_L2 as dataset
from sklearn import metrics
import network_L3 as network
import os
from FocalLoss import FocalLoss
import time
import torch.nn.functional as F

start = time.time()

# 参数设置
PATH = 'data/labels.csv'  # real_images_256  labels
TEST_PATH = 'data/exam_labels.csv'
is_train = False  # True-训练模型  False-测试模型
backbone = 'resnet18'  # 骨干网络：alexnet resnet18 vgg16 densenet inception
save_model_name = 'model/' + backbone + 'bSMOTE2.pkl'  # 模型存储路径
# save_model_name = 'model/L2_model.pkl'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_pretrained = True  # 是否加载预训练权重
is_sampling = 'over_sampler'  # 训练集采样模式： over_sampler-上采样  down_sampler-下采样  no_sampler-无采样
print('Device:', device)

# 训练参数设置
SIZE = 299 if backbone == 'inception' else 224
BATCH_SIZE = 32  # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 10  # 迭代次数

# 进入工程路径并新建文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 进入工程路径
dataset.mkdir('model')  # 新建文件夹

# 加载数据
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train,
                                                            is_sampling=is_sampling)

# 定义模型、优化器、损失函数
model = network.initialize_model(backbone, is_pretrained)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=6e-3)  # 更新所有层权重
# criterion = torch.nn.CrossEntropyLoss()
criterion = FocalLoss()


# 训练模型
def train_alexnet(model):
    for epoch in range(EPOCHS):
        correct = total = 0.
        loss_list = []
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            model.train()
            # 优化过程
            optimizer.zero_grad()  # 梯度归0
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            # 输出训练结果
            loss_list.append(loss.item())
            _, predicted = torch.max(output.data, 1)  # 返回每行的最大值
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        train_avg_acc = 100 * correct / total
        train_avg_loss = np.mean(loss_list)
        print('[Epoch=%d/%d]Train set: Avg_loss=%.4f, Avg_accuracy=%.4f%%' % (
            epoch + 1, EPOCHS, train_avg_loss, train_avg_acc))

    # 保存模型
    torch.save(model.state_dict(), save_model_name)
    print('Training finished!')

def for_test_resnet(best_model_name):
    print('------ Testing Start ------')
    model.load_state_dict(torch.load(best_model_name), False)
    test_pred = []
    test_true = []
    test_prob = np.empty(shape=[0, 2])  # 概率值

    with torch.no_grad():
        model.eval()
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            else:
                images, labels = test_x, test_y
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            prob = F.softmax(output.data, dim=1)  # softmax[[0.9,0.1],[0.8,0.2]]
            test_prob = np.append(test_prob, prob.detach().cpu().numpy(), axis=0)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    images = test_loader.dataset.test_img
    test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    test_AUC = metrics.roc_auc_score(y_true=test_true, y_score=test_prob[:, 1])  # y_score=正例的概率
    # test_AUC = metrics.roc_auc_score(y_true=test_true, y_score=test_pred)
    test_classification_report = metrics.classification_report(test_true, test_pred, digits=4)
    tn, fp, fn, tp = metrics.confusion_matrix(test_true, test_pred).ravel()
    print('test_classification_report\n', test_classification_report)
    print('Accuracy of the network is: %.4f %%' % test_acc)
    print('Test_AUC: %.4f' % test_AUC)
    print('TN=%d, FP=%d, FN=%d, TP=%d' % (tn, fp, fn, tp))
    return test_acc, images, test_true, test_pred


if __name__ == '__main__':
    if is_train:
        train_alexnet(model)
    else:
        test_acc, test_img, test_true, test_pred = for_test_resnet(save_model_name)
        show_batch = 100
        iters = math.ceil(test_img.shape[0] / show_batch)
        begin = 0
        for iter in range(iters):
            end = begin + show_batch if (begin + show_batch) <= test_img.shape[0] else test_img.shape[0]
            show_test_img, show_test_true, show_test_pred = test_img[begin:end], test_true[begin:end], test_pred[
                                                                                                       begin:end]
            dataset.show_test(show_test_img, show_test_true, show_test_pred, show_batch, iter)
            begin = end
            end = begin + show_batch
            plt.savefig('{}/{}.jpg'.format("./model", (str(iter + 1).rjust(4, '0'))))

    print('头部ct运行时间level2：', time.time() - start)
