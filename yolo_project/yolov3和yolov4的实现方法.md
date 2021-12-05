# yolov3和yolov4的实现方法

## 摘要

本文是基于github开源项目darknet实现简易Object detection的搭建方法，算法系统主要实现于yolov3和yolov4，搭建平台为ubuntu20.04(wsl)

注：yolov4已经能够实现在window下实现，但yolov3还是只能够在linux下搭建

本文主要测试yolov4的检测方法

## 项目介绍

yolo算法是目前较为先进的目标检测算法系统，目前已更新至第五代

官方论文介绍到：

> 我们呈现YOLO，一种对象检测的新方法。 在对象检测的事前进行处理重新调整分类器进行检测。 相反，我们将对象检测作为空间分离的边界框和相关类概率的回归问题。 单个神经网络在一个评估中直接从完整图像预测边界框和类概率。 由于整个检测管道是单个网络，因此可以直接在检测性能上结束端到端。
>
> 我们统一的架构非常快。 我们的基础YOLO模型实时处理图像每秒45帧。 较小版本的网络，快速yolo，每秒处理惊人的155帧，同时仍然实现了其他实时探测器的地图。 与最先进的检测系统相比，YOLO制造了更多的本地化误差，但无法预测任何没有存在的错误检测。 最后，YOLO了解对象的非常一般的表示。 当从自然图像到毕加索数据集和人民艺术数据集的自然图像到艺术品时，它占据了所有其他检测方法，包括DPM和R-CNN的所有其他检测方法，包括DPM和R-CNN。

## reference：

https://github.com/AlexeyAB/darknet （yolov4）

https://github.com/ultralytics/yolov3

## 前提

首先确保ubuntu已具备一下条件:

- opencv的编译
- cmake的编译
- 同时为了加快识别速度，开启了gpu加速，这就需要安装cuda和cudnn
- ubuntu中相关系统库的安装

## 编译实现

### 从github上克隆项目

```
git clone https://github.com/AlexeyAB/darknet.git
```

### 准备编译

a、进入到darknet目录下，输入

```make
make
```

**注：如果报错缺少相关包，首先应检测自己的ubuntu是不是缺少相关支持包，如果检查没有缺少的可以换一种方法编译**

------

b、创建build_release,然后重新编译，依次输入

```make
mkdir build_release
cd build_release
cmake ..
cmake --build . --target install parallel 8
```

**注：如果直接在这里输入检测目录会报错无法打开相关文件**

**解决办法：将编译生成的darknet可执行文件移动到克隆好的根目录下（即darknet下）**

### 检验测试

输入指令 （注：在输入之前提前将yolov4的预训练的权重yolov4.weights放在克隆好的darknet目录下）

```make
./darknet detect cfg/yolov4.cfg yolov4.weights data/dog.jpg
```

### 检测结果

#### 实例1：

对本地图片进行检测（注：要检测图片时需要将图片放置到data文件下，也可指定位置）

<img src="https://raw.githubusercontent.com/Richard-LiSR/PicBed/master/image-20210629110144323.png" alt="image-20210629110144323"  />

**从上图中可以清晰的发现，yolov4已经能够准确的检测到 dog、bicycle 、truck、potteplant四种物体，且准确度都较高。**



#### 实例2：

对视频流进行实时检测

编写可执行文件video_yolov4.sh中视频的路径

```ma
./darknet detector demo ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights data/race.mp4 -i 0 -thresh 0.25
```

编写完成后保存，输入

```ma
./video_yolov4.sh
```

<img src="https://raw.githubusercontent.com/Richard-LiSR/PicBed/master/image-20210629111208663.png" alt="image-20210629111208663" style="zoom: 67%;" />



**yolov3上同**
