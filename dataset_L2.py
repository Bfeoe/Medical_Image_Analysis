import math

from pylab import *
import pydicom
import cv2
import os
from sklearn.model_selection import train_test_split
import sklearn.utils
from collections import Counter
import torch.utils.data as data
import torchvision.transforms as transforms

# -------------------------------#
#            参数设置
# -------------------------------#
random_seed = 321  # 随机种子
ratio = 0.1  # 验证集、测试集比例


# -------------------------------#
#          图像预处理方法
# -------------------------------#

# 可选：图像阈值范围缩减
def windows_pro(img, min_bound=0, max_bound=85):
    """
        输入：图像，阈值下限min_bound，阈值上限max_bound
        处理过程：先获取指定限制范围内的值[min_bound,max_bound]，再中心化、归一化
        输出：阈值范围缩减后中心化归一化结果[0,255]
    """
    img[img > max_bound] = max_bound
    img[img < min_bound] = min_bound  # [min_bound, max_bound]
    img = img - min_bound  # 中心化[0,max_bound+min_bound]
    img = normalize(img)  # 归一化 [0,255]
    return img


# 可选：直方图均衡(增加对比度)
def equalize_hist(img):
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    return img


# 必选：缩放尺寸，默认缩放为224
def img_resize(img, size=224):
    img = cv2.resize(img, (size, size))
    return img


# 必选：归一化
def normalize(img):
    img = img.astype(np.float32)
    np.seterr(divide='ignore', invalid='ignore')
    img = (img - img.min()) / (img.max() - img.min())  # 归一化[0,1]
    img = img * 255  # 0-255
    img = img.astype(np.uint8)
    return img


# 必选：扩展为3通道
def extend_channels(img):
    img_channels = np.zeros([img.shape[0], img.shape[1], 3])
    img_channels[:, :, 0] = img
    img_channels[:, :, 1] = img
    img_channels[:, :, 2] = img
    return img_channels


# 必选：图像预处理组合（基本操作）
def data_preprocess_base(img, size):
    # step1: 缩放尺寸 224*224
    img = img_resize(img, size)
    # step2: 归一化[0,255]
    img = normalize(img)
    # step3: 扩展为3通道 224*224*3
    img = extend_channels(img)
    # Step4: 转换为unit8格式
    img = img.astype(np.uint8)
    return img


# 图像预处理（伪影增强）
def data_preprocess_enhanced(img, size):
    # step1: 图像阈值范围缩减 [min_bound, max_bound]
    img = windows_pro(img)
    # step2: 直方图均衡 [0, 255]
    img = equalize_hist(img)
    # step3: 缩放尺寸 224*224
    img = img_resize(img, size)
    # step4: 归一化[0,255]
    img = normalize(img)
    # step5: 扩展为3通道 224*224*3
    img = extend_channels(img)
    # Step6: 转换为unit8格式
    img = img.astype(np.uint8)
    return img


# -------------------------------#
#       读取数据并划分数据集
# -------------------------------#
def data_load(path, test_path, size, is_train, is_sampling='no_sampler'):
    dicomlist = []    # 图像地址
    labels = []       # 图像标签
    train_img = []    # 训练集图像
    train_label = []  # 训练集标签
    val_img = []      # 验证集图像
    val_label = []    # 验证集标签


    # 1.读取数据：images图像矩阵，labels标签
    f = open(path, "r+") if test_path == '' else open(test_path, "r+")
    for line in f.readlines():
        img_path = line.strip().split(',')[0] # 图像地址
        dicomlist.append(img_path)
        label = line.strip().split(',')[1]  # 图像标签
        label = '0' if label == 'good' else '1'
        labels.append(label)
    labels = np.array(labels)  # 图像标签 n*1
    # 读取图像矩阵
    images = array([data_preprocess_enhanced(pydicom.read_file(dcm).pixel_array, size) for dcm in dicomlist])
    f.close()

    # 2.划分数据集
    if is_train or test_path == '':  # 训练模式或测试模式没有单独csv
        print('----Training Mode----') if is_train else print('----Testing mode----')
        # 划分数据集：训练集、验证集、测试集
        images, labels = sklearn.utils.shuffle(images, labels, random_state=random_seed)  # images = n*224*224
        train_val_img, test_img, train_val_label, test_label = train_test_split(images, labels, test_size=ratio,
                                                                        stratify=labels,random_state=random_seed)

        train_img, val_img, train_label, val_label = train_test_split(train_val_img, train_val_label,
                                                    test_size=ratio,stratify=train_val_label, random_state=random_seed)

        print('Dataset: %s, labels=%s' % (images.shape, sorted(Counter(labels).items())))
        print('Training set: %s, labels=%s' % (train_img.shape, sorted(Counter(train_label).items())))
        print('Val set: %s, labels=%s' % (val_img.shape, sorted(Counter(val_label).items())))
        print('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))

        # 训练集采样模式
        if is_train:
            if is_sampling == 'no_sampler':
                pass
            elif is_sampling == 'over_sampler':
                train_img, train_label = over_sampling(train_img, train_label)
            elif is_sampling == 'down_sampler':
                train_img, train_label = under_sampling(train_img, train_label)
            print('Sampling mode:%s, train_num:%s,label:%s' % (
                is_sampling, train_img.shape, sorted(Counter(train_label).items())))

    else:  # 测试模式
        print('----Testing Mode----')
        test_img = images
        test_label = labels
        print('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))
    return train_img, train_label, val_img, val_label, test_img, test_label


class TrainDataset(data.Dataset):
    def __init__(self, train_img, train_label, train_data_transform=None):
        super(TrainDataset, self).__init__()
        self.train_img = train_img
        self.train_label = train_label
        self.train_data_transform = train_data_transform

    def __getitem__(self, index):
        img = self.train_img[index]
        target = int(self.train_label[index])
        if self.train_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))  # narray->PIL
            img = self.train_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.train_img)


class ValDataset(data.Dataset):
    def __init__(self, val_img, val_label, val_data_transform):
        super(ValDataset, self).__init__()
        self.val_img = val_img
        self.val_label = val_label
        self.val_data_transform = val_data_transform

    def __getitem__(self, index):
        img = self.val_img[index]
        target = int(self.val_label[index])
        if self.val_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))  # narray->PIL
            img = self.val_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.val_img)


class TestDataset(data.Dataset):
    def __init__(self, test_img, test_label, test_data_transform):
        super(TestDataset, self).__init__()
        self.test_img = test_img
        self.test_label = test_label
        self.test_data_transform = test_data_transform

    def __getitem__(self, index):
        img = self.test_img[index]
        target = int(self.test_label[index])
        if self.test_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            img = self.test_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.test_img)


# -------------------------------#
#          加载数据集
# -------------------------------#
def get_dataset(path, test_path, size, batch_size, is_train, is_sampling=False):
    train_img, train_label, val_img, val_label, test_img, test_label = data_load(path, test_path, size, is_train,
                                                                                 is_sampling)
    train_loader = []
    val_loader = []

    if is_train:
        # 定义train_loader
        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 180), expand=False),
            transforms.ToTensor()])

        train_set = TrainDataset(train_img, train_label, train_data_transform)
        train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        # 定义val_loader
        val_data_transform = transforms.Compose([
            transforms.ToTensor()])
        val_set = ValDataset(val_img, val_label, val_data_transform)
        val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # 定义test_loader
    test_data_transform = transforms.Compose([
        transforms.ToTensor()])
    test_set = TestDataset(test_img, test_label, test_data_transform)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ----------------------------------------------------#
#    训练集正负样本不均衡采样方式：下采样、上采样
# ----------------------------------------------------#
# 下采样
# def under_sampling(train_img, train_label):
#     from imblearn.under_sampling import RandomUnderSampler
#     rus = RandomUnderSampler(random_state=random_seed, replacement=False)
#     nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
#     train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
#     X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
#     X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
#     return X_resampled, y_resampled

# def under_sampling(train_img, train_label):
#     from imblearn.under_sampling import NearMiss
#     rus = NearMiss(version=3)
#     nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
#     train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
#     X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
#     X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
#     return X_resampled, y_resampled

# def under_sampling(train_img, train_label):
#     from imblearn.under_sampling import CondensedNearestNeighbour
#     rus = CondensedNearestNeighbour(n_neighbors=5)
#     nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
#     train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
#     X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
#     X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
#     return X_resampled, y_resampled

def under_sampling(train_img, train_label):
    from imblearn.under_sampling import OneSidedSelection
    rus = OneSidedSelection(n_neighbors=3, n_seeds_S=random_seed)
    nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled
# 上采样
# def over_sampling(train_img, train_label):
#     from imblearn.over_sampling import RandomOverSampler
#     rus = RandomOverSampler(random_state=random_seed)
#     nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
#     train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
#     X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
#     X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
#     return X_resampled, y_resampled

# def over_sampling(train_img, train_label):
#     from imblearn.over_sampling import SMOTE
#     rus = SMOTE(k_neighbors=2, random_state=random_seed)
#     nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
#     train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
#     X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
#     X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
#     return X_resampled, y_resampled

def over_sampling(train_img, train_label):
    from imblearn.over_sampling import BorderlineSMOTE
    rus = BorderlineSMOTE(k_neighbors=2, random_state=random_seed)
    nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled


# 绘制loss、accuracy、AUC
def show_plot(history_train, history_valid, history_auc, model_path):
    # 绘制训练集和验证集的损失值
    x = range(0, len(np.array(history_auc)))
    plt.figure(1)  # 第一张图
    plt.plot(x, np.array(history_train)[:, 0])
    plt.plot(x, np.array(history_valid)[:, 0])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    # plt.show()
    plt.savefig('{}/{}.jpg'.format(model_path, 'Model Loss'))

    # 绘制训练集和验证集的精确度
    plt.figure(2)  # 第二张图
    plt.plot(x, np.array(history_train)[:, 1])
    plt.plot(x, np.array(history_valid)[:, 1])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    # plt.show()
    plt.savefig('{}/{}.jpg'.format(model_path, 'Model Accuracy'))

    # 绘制验证集AUC
    plt.figure(3)  # 第三张图
    plt.plot(x, np.array(history_auc))
    plt.title('Validation AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.savefig('{}/{}.jpg'.format(model_path, 'Validation AUC'))



def show_test(test_img, test_true, test_pred, show_batch, iter):
    images = test_img
    true = test_true
    predict = test_pred
    image_number = images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number + 8, column_number + 8))

    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index + 1)
                image = images[index].astype('uint8')

                '''
                # 测试集无标签
                colour = 'red' if '1' in str(predict[index]) else 'blue'
                tag = 'bad' if '1' in str(predict[index]) else 'good'
                title = 'id:%s,predict:%s' % (str(index + 1 + show_batch * iter), str(tag))
                '''

                # 测试集有标签
                if true[index] == predict[index]:
                    colour = 'black'
                else:
                    colour = 'red' if '1' in str(true[index]) else 'blue'
                tag_pred = 'bad' if '1' in str(predict[index]) else 'good'
                tag_true = 'bad' if '1' in str(true[index]) else 'good'
                title = 'id:%s,true:%s,pred:%s' % (str(index + 1 + show_batch * iter), str(tag_true), str(tag_pred))

                plt.subplot(*position)
                plt.imshow(image, cmap='gray_r')  # 3-channel
                plt.axis('off')
                plt.title(title, fontsize=8, color=colour)


# 新建文件夹
def mkdir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def show_curve(model_path, test_true, test_pred, test_prob):
    """
    绘制单模型ROC、P-R、Confusion Matrix
    """
    # ROC曲线
    import matplotlib.pyplot as plt
    import scikitplot as skplt
    skplt.metrics.plot_roc(test_true, test_prob)
    plt.savefig('{}/{}.jpg'.format(model_path, 'ROC Curve'))
    plt.show()

    # PR曲线
    skplt.metrics.plot_precision_recall_curve(test_true, test_prob, cmap='nipy_spectral')
    plt.savefig('{}/{}.jpg'.format(model_path, 'P-R Curve'))
    plt.show()

    # Confusion Matrix
    skplt.metrics.plot_confusion_matrix(test_true, test_pred, normalize=False)
    plt.savefig('{}/{}.jpg'.format(model_path, 'confusion_matrix'))
    plt.show()


# 绘制多模型ROC曲线图
def show_multi_roc(model_path, model_list, true_list, prob_list):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle

    # 画平均ROC曲线的两个参数
    mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    mean_fpr = np.linspace(0, 1, 100)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])  # 颜色
    lw = 2  # 粗细
    cnt = 0
    for model_name, true, probas_, color in zip(model_list, true_list, prob_list, colors):
        cnt += 1
        fpr, tpr, thresholds = roc_curve(true, probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点
        roc_auc = auc(fpr, tpr)  # 求auc面积
        plt.plot(fpr, tpr, color=color, lw=lw,
                 label='{0} (AUC = {1:.3f})'.format(model_name, roc_auc))  # 画出当前分割数据的ROC曲线

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Reference')  # 画对角线

    mean_tpr /= cnt  # 求数组的平均值
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = {0:.3f})'.format(mean_auc), lw=2)

    plt.xlim([0.00, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([0.00, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-ROC Curves: External test set')
    plt.legend(loc="lower right")
    plt.savefig('{}/{}.jpg'.format(model_path, 'Multi-ROC Curves: test set'))
    plt.show()
