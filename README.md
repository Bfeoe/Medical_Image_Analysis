# 医疗影像分析
## 1.CT医学影像伪影识别

首先，``通过百度网盘分享的文件：checkpoint_best_1.pkl
链接：https://pan.baidu.com/s/1KabzESMMmS7J7phYmHL1Fw?pwd=mswi 
提取码：mswi``
中下载（我的Git LFS不知出了什么问题，抱歉）我们训练好的模型文件： ``checkpoint_best_1.pkl``

将下载好的模型文件置于 ``CT_Classification/pre_model`` 下。

然后，运行 ``CT_Classification/test_L2.py`` 文件

## 2.超声左心室和结肠息肉分割

首先，``通过百度网盘分享的文件：checkpoint_best.pth
链接：https://pan.baidu.com/s/12f_Gq880ZgXhqtDmZPLvKw?pwd=mswi 
提取码：mswi``

下载的我们训练好的模型文件： ``checkpoint_best.pth``
还有我们整合的数据文件： ``通过百度网盘分享的文件：union.zip
链接：https://pan.baidu.com/s/1v3xXnTdvnYY8-c6UiVj0Hw?pwd=mswi 
提取码：mswi``

将下载好的模型文件置于 ``finetune-SAM/2D-SAM_vit_b_decoder_adapter_TrainDataset_noprompt`` 下,
将下载好的数据文件置于 ``finetune-SAM`` 下

运行 ``finetune-SAM/2D_predictions_with_vis.ipynb`` 即可进行预测并可视化结果
（修改image_path与ground_truth_path对应的图片编号，即可修改对应要预测的图片）
