# 医疗影像分析
## 1.CT医学影像伪影识别

首先，下载 ``release`` 中的我们训练好的模型文件： ````

将下载好的模型文件置于 ``CT_Classification/pre_model`` 下。

然后，运行 ``CT_Classification/test_L2.py`` 文件

## 2.超声左心室和结肠息肉分割

首先，下载 ``release`` 中的我们训练好的模型文件： ``checkpoint_best.pth``

将下载好的模型文件置于 ``finetune-SAM/2D-SAM_vit_b_decoder_adapter_TrainDataset_noprompt`` 下

运行 ``finetune-SAM/2D_predictions_with_vis.ipynb`` 即可进行预测并可视化结果
（修改image_path与ground_truth_path对应的图片编号，即可修改对应要预测的图片）
