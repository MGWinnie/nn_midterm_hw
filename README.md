# nn_midterm_hw
## 图像分类任务
#### 训练
```python 
python Train.py -net resnet18 -gpu -lr 0.1 -method cutout/cutmix/mixup/none
```
#### 结果
训练好的模型参数存入checkpoint文件夹下，tensorboard所画曲线存入runs文件夹下

## 目标检测任务
### Faster R-CNN
#### 训练
将backbone模型参数resnet50.pth和预训练模型参数fasterrcnn_resnet50_fpn_coco.pth添加至./backbone，并运行train_res50_fpn.py
#### 测试
将训练好的模型参数rcnn_model_weights.pth添加至./save_weights，并运行predict.py
### YOLO V3
