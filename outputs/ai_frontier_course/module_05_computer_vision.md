# 模块五：计算机视觉前沿技术

## 课程信息
- **模块编号**：Module 05
- **模块名称**：计算机视觉前沿技术
- **学习目标**：掌握现代计算机视觉的核心技术、前沿模型和实际应用
- **学时安排**：理论课程 14学时，实践课程 12学时

## 第一章：计算机视觉基础

### 1.1 图像处理基础

#### 数字图像表示

**什么是数字图像？**

想象一下，计算机是如何"看"图片的？其实很简单，就像我们用马赛克拼图一样！

**像素：图像的基本单位**

🎯 **简单理解**：像素就是图像的"小方块"，就像乐高积木的最小单元

📝 **详细说明**：
- 每个像素都有一个位置坐标(x,y)
- 每个像素都有颜色信息
- 像素越多，图像越清晰（就像马赛克块越小，拼出的图越精细）

**颜色空间：不同的"调色盘"**

*灰度图像*
🎯 **简单理解**：就像黑白照片，只有明暗变化
📝 **详细说明**：
- 每个像素只有一个数值，表示亮度
- 0表示纯黑，255表示纯白
- 中间的数值表示不同程度的灰色
🌰 **举例**：老式黑白电视、X光片、报纸照片

*彩色图像*
🎯 **简单理解**：就像调色盘，用红绿蓝三种颜色混合出所有颜色
📝 **详细说明**：
- **RGB模式**：像三盏彩色灯（红Red、绿Green、蓝Blue）
  - 三盏灯都不亮 = 黑色
  - 三盏灯都最亮 = 白色
  - 不同亮度组合 = 各种颜色
- **HSV模式**：更接近人眼感知
  - H（色调）：什么颜色？红色还是蓝色？
  - S（饱和度）：颜色有多鲜艳？
  - V（明度）：颜色有多亮？
🌰 **举例**：
- 电脑屏幕用RGB显示
- 画家调色更像HSV思维

**图像变换：让图片"动起来"**

想象你在手机上编辑照片，那些旋转、缩放、调亮度的操作，其实就是图像变换！

*几何变换：改变图片的形状和位置*

🎯 **简单理解**：就像在纸上移动、旋转、放大缩小一张贴纸

📝 **详细说明**：
- **平移（Translation）**：把图片整体移动
  - 就像把照片从桌子左边移到右边
  - 图片内容不变，只是位置变了
  
- **旋转（Rotation）**：把图片转个角度
  - 就像把手机横过来看照片
  - 围绕某个点转动
  
- **缩放（Scaling）**：把图片放大或缩小
  - 就像用手指在手机上"捏"照片
  - 可以等比例缩放，也可以只改变宽度或高度

🌰 **举例**：
- 证件照调整位置和大小
- 地图的缩放和旋转
- 游戏中角色的移动和转向

*强度变换：改变图片的亮度和对比度*

🎯 **简单理解**：就像调节电视的亮度、对比度按钮

📝 **详细说明**：
- **线性变换**：统一调整亮度和对比度
  - 就像给所有像素加上或减去一个固定值
  - 让图片整体变亮或变暗
  
- **伽马校正**：非线性调整
  - 就像调节显示器的"伽马值"
  - 让暗部细节更清晰，或让亮部不过曝
  
- **直方图均衡化**：自动优化对比度
  - 就像相机的"自动曝光"
  - 让图片的明暗分布更均匀

🌰 **举例**：
- 手机拍照后的自动美化
- 老照片的修复和增强
- 医学影像的对比度调整

#### 滤波与特征提取

**什么是滤波？**

🎯 **简单理解**：就像给图片戴上"特殊眼镜"，让我们看到不同的效果

想象一下，你戴上墨镜看世界会变暗，戴上有色眼镜看世界会变色。图像滤波就是给图片戴上各种"数字眼镜"！

**空域滤波：直接在图片上"动手术"**

*线性滤波：用"模板"处理图片*

🎯 **简单理解**：就像用印章在图片上盖印，每个位置都用同样的"模板"处理

📝 **详细说明**：
- 拿一个小"窗口"（滤波器核）在图片上滑动
- 每到一个位置，就按照窗口的"规则"计算新的像素值
- 就像用九宫格模板，把周围9个像素按权重加起来

*常用滤波器及其效果*

**均值滤波器："模糊"效果**
🎯 **简单理解**：就像近视眼看东西，让图片变模糊
📝 **详细说明**：
- 把每个像素和周围像素的平均值作为新值
- 可以去除噪点，但会让图片变模糊
🌰 **举例**：老照片去噪、皮肤磨皮效果

**高斯滤波器："柔和模糊"效果**
🎯 **简单理解**：就像专业摄影的"柔焦"效果
📝 **详细说明**：
- 比均值滤波更自然的模糊效果
- 中心权重大，边缘权重小
- 既能去噪又能保持图片的自然感
🌰 **举例**：人像摄影的背景虚化、图片预处理

**Sobel边缘检测："轮廓提取"效果**
🎯 **简单理解**：就像用铅笔描出物体的轮廓线
📝 **详细说明**：
- 专门找图片中亮度变化剧烈的地方
- 可以检测水平和垂直方向的边缘
- 把物体的形状"画"出来
🌰 **举例**：
- 自动驾驶识别道路边缘
- 医学影像中器官轮廓提取
- 工业检测中零件边缘识别

**频域滤波**

*傅里叶变换*
```
F(u,v) = ∑∑ f(x,y)exp(-j2π(ux/M + vy/N))
```
- 频域表示
- 低通、高通、带通滤波
- 快速傅里叶变换（FFT）

*频域滤波器*
- 理想低通滤波器
- 巴特沃斯滤波器
- 高斯滤波器

#### 特征描述子

**边缘检测**

*Canny边缘检测*
1. 高斯滤波去噪
2. 计算梯度幅值和方向
3. 非极大值抑制
4. 双阈值检测和连接

*Laplacian算子*
```
∇²f = ∂²f/∂x² + ∂²f/∂y²

离散形式:
[0 -1  0]
[-1 4 -1]
[0 -1  0]
```

**角点检测**

*Harris角点检测*
```
M = ∑ w(x,y) [Ix²   IxIy]
              [IxIy  Iy² ]

R = det(M) - k(trace(M))²
```
- Ix, Iy：图像梯度
- w(x,y)：窗口函数
- k：经验常数（0.04-0.06）

*FAST角点检测*
- Features from Accelerated Segment Test
- 基于像素强度比较
- 计算效率高
- 适合实时应用

**局部特征描述**

*SIFT（Scale-Invariant Feature Transform）*
1. 尺度空间极值检测
2. 关键点定位
3. 方向分配
4. 特征描述子生成

*SURF（Speeded Up Robust Features）*
- SIFT的快速近似
- 使用积分图像
- Hessian矩阵检测
- 64维或128维描述子

*ORB（Oriented FAST and Rotated BRIEF）*
- 结合FAST和BRIEF
- 旋转不变性
- 二进制描述子
- 计算效率高

### 1.2 传统机器学习方法

#### 特征工程

**手工特征设计**

*纹理特征*
- 灰度共生矩阵（GLCM）
- 局部二值模式（LBP）
- Gabor滤波器响应
- 小波变换系数

*形状特征*
- 轮廓描述子
- 傅里叶描述子
- 不变矩
- 链码

*颜色特征*
- 颜色直方图
- 颜色矩
- 颜色相关图
- 主色彩

**特征选择与降维**

*特征选择*
- 过滤方法：相关性分析
- 包装方法：前向/后向选择
- 嵌入方法：L1正则化

*降维技术*
- 主成分分析（PCA）
- 线性判别分析（LDA）
- 独立成分分析（ICA）
- t-SNE可视化

#### 分类器设计

**支持向量机（SVM）**

*线性SVM*
```
f(x) = w^T x + b
目标: min (1/2)||w||² + C∑ξᵢ
约束: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ
```

*核SVM*
```
f(x) = ∑ αᵢyᵢK(xᵢ,x) + b

常用核函数:
线性核: K(x,x') = x^T x'
多项式核: K(x,x') = (γx^T x' + r)^d
高斯核: K(x,x') = exp(-γ||x-x'||²)
```

**随机森林**
- 集成学习方法
- Bootstrap采样
- 随机特征选择
- 投票决策

**AdaBoost**
```
H(x) = sign(∑ αₜhₜ(x))

αₜ = (1/2)ln((1-εₜ)/εₜ)
Dₜ₊₁(i) = Dₜ(i)exp(-αₜyᵢhₜ(xᵢ))/Zₜ
```

#### 目标检测经典方法

**滑动窗口**
- 多尺度窗口扫描
- 特征提取和分类
- 非极大值抑制
- 计算复杂度高

**Viola-Jones检测器**

*Haar特征*
```
矩形特征:
□■  ■□  □■□  ■□■
■□  □■  ■□■  □■□
```

*积分图像*
```
ii(x,y) = ∑ᵢ≤ₓ ∑ⱼ≤ᵧ I(i,j)

矩形和 = ii(D) + ii(A) - ii(B) - ii(C)
```

*AdaBoost级联*
- 强分类器级联
- 早期拒绝
- 实时人脸检测

**HOG + SVM**

*方向梯度直方图*
1. 梯度计算
2. 方向量化（9个bin）
3. 空间分块（8×8像素）
4. 块归一化（2×2块）
5. 特征向量连接

*行人检测*
- 64×128窗口
- 3780维HOG特征
- 线性SVM分类
- 多尺度检测

### 1.3 深度学习革命

#### CNN架构演进

**LeNet-5（1998）**
```
输入: 32×32灰度图像
C1: 6个5×5卷积核
S2: 2×2平均池化
C3: 16个5×5卷积核
S4: 2×2平均池化
C5: 120个5×5卷积核
F6: 84个全连接
输出: 10个全连接
```
- 卷积+池化的经典模式
- 手写数字识别
- 现代CNN的鼻祖

**AlexNet（2012）**

*创新点*
- ReLU激活函数
- Dropout正则化
- 数据增强
- GPU并行训练
- 局部响应归一化（LRN）

*网络结构*
```
输入: 224×224×3
Conv1: 96×11×11, stride=4, pad=0
MaxPool1: 3×3, stride=2
Conv2: 256×5×5, stride=1, pad=2
MaxPool2: 3×3, stride=2
Conv3: 384×3×3, stride=1, pad=1
Conv4: 384×3×3, stride=1, pad=1
Conv5: 256×3×3, stride=1, pad=1
MaxPool3: 3×3, stride=2
FC1: 4096
FC2: 4096
FC3: 1000
```

**VGGNet（2014）**

*设计原则*
- 小卷积核（3×3）
- 深度网络（16-19层）
- 简洁统一的结构

*VGG-16结构*
```
Block1: 2×Conv(64,3×3) + MaxPool
Block2: 2×Conv(128,3×3) + MaxPool
Block3: 3×Conv(256,3×3) + MaxPool
Block4: 3×Conv(512,3×3) + MaxPool
Block5: 3×Conv(512,3×3) + MaxPool
FC: 4096 → 4096 → 1000
```

*优势*
- 证明深度的重要性
- 小卷积核的有效性
- 预训练模型广泛使用

**ResNet（2015）**

*残差学习*
```
y = F(x, {Wᵢ}) + x
```
- 恒等映射快捷连接
- 解决梯度消失问题
- 训练超深网络（152层）

*残差块设计*
```
基本块:
x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+x) → ReLU

瓶颈块:
x → Conv(1×1) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → BN → (+x) → ReLU
```

*网络变体*
- ResNet-18/34：基本块
- ResNet-50/101/152：瓶颈块
- ResNeXt：分组卷积
- Wide ResNet：增加宽度

#### 现代CNN架构

**Inception系列**

*Inception v1 (GoogLeNet)*
```
Inception模块:
1×1 conv → output
1×1 conv → 3×3 conv → output
1×1 conv → 5×5 conv → output
3×3 maxpool → 1×1 conv → output
↓
Concatenate
```
- 多尺度特征提取
- 1×1卷积降维
- 22层深度

*Inception v2/v3*
- 批归一化
- 分解卷积（5×5 → 两个3×3）
- 非对称卷积（7×1和1×7）
- 辅助分类器

*Inception v4*
- 结合ResNet思想
- 更深更宽的网络
- 统一的stem结构

**DenseNet（2017）**

*密集连接*
```
xₗ = Hₗ([x₀, x₁, ..., xₗ₋₁])
```
- 每层连接到所有前面的层
- 特征重用
- 参数效率高

*密集块*
```
BN → ReLU → Conv(1×1) → BN → ReLU → Conv(3×3)
```
- 增长率k（每层增加k个特征图）
- 过渡层（1×1卷积+池化）
- 压缩因子θ

**EfficientNet（2019）**

*复合缩放*
```
depth: d = α^φ
width: w = β^φ  
resolution: r = γ^φ
约束: α·β²·γ² ≈ 2, α≥1, β≥1, γ≥1
```
- 同时缩放深度、宽度、分辨率
- 神经架构搜索（NAS）
- 参数效率和精度的最佳平衡

*MBConv块*
```
输入 → 1×1扩展卷积 → DW卷积 → SE模块 → 1×1投影卷积 → 输出
```
- 移动端优化的倒残差结构
- Squeeze-and-Excitation注意力
- 随机深度正则化

## 第二章：目标检测与分割

### 2.1 目标检测

#### 两阶段检测器

**R-CNN（2014）**

*算法流程*
1. 选择性搜索生成候选区域（~2000个）
2. CNN特征提取（AlexNet）
3. SVM分类
4. 边界框回归

*问题*
- 计算冗余（每个候选区域独立CNN）
- 训练复杂（多阶段）
- 速度慢（~47秒/图）

**Fast R-CNN（2015）**

*改进*
- 整图CNN特征提取
- ROI池化层
- 多任务损失（分类+回归）
- 端到端训练

*ROI池化*
```
将任意大小的ROI映射到固定大小的特征图
ROI: (x, y, w, h) → 固定大小: H×W
```

*多任务损失*
```
L = L_cls + λL_bbox
L_cls = -log p_u  (u为真实类别)
L_bbox = smooth_L1(t_u - v)
```

**Faster R-CNN（2017）**

*区域提议网络（RPN）*
```
输入: 卷积特征图
输出: 目标性分数 + 边界框回归
```

*锚点机制*
- 每个位置k个锚点
- 多尺度（128², 256², 512²）
- 多长宽比（1:1, 1:2, 2:1）

*RPN损失*
```
L_RPN = (1/N_cls)∑L_cls(pᵢ, pᵢ*) + λ(1/N_reg)∑pᵢ*L_reg(tᵢ, tᵢ*)
```
- pᵢ：预测目标性概率
- pᵢ*：真实标签（0或1）
- tᵢ：预测边界框
- tᵢ*：真实边界框

*四步训练*
1. 训练RPN
2. 训练Fast R-CNN（使用RPN提议）
3. 微调RPN（固定共享层）
4. 微调Fast R-CNN（固定共享层）

#### 单阶段检测器

**YOLO（You Only Look Once）**

*基本思想*
- 将检测转化为回归问题
- 整图一次前向传播
- 实时检测

*YOLOv1算法*
1. 图像分割为S×S网格
2. 每个网格预测B个边界框
3. 每个边界框预测5个值：(x,y,w,h,confidence)
4. 每个网格预测C个类别概率

*输出张量*
```
S×S×(B×5 + C)
例如: 7×7×(2×5 + 20) = 7×7×30
```

*损失函数*
```
L = λ_coord∑∑∑[1_ij^obj][(xᵢ-x̂ᵢ)² + (yᵢ-ŷᵢ)²]
  + λ_coord∑∑∑[1_ij^obj][(√wᵢ-√ŵᵢ)² + (√hᵢ-√ĥᵢ)²]
  + ∑∑∑[1_ij^obj][(Cᵢ-Ĉᵢ)²]
  + λ_noobj∑∑∑[1_ij^noobj][(Cᵢ-Ĉᵢ)²]
  + ∑∑[1_i^obj]∑[(pᵢ(c)-p̂ᵢ(c))²]
```

**YOLOv2/YOLO9000**

*改进*
- 批归一化
- 高分辨率分类器
- 锚点机制
- 维度聚类
- 直接位置预测
- 细粒度特征
- 多尺度训练

*Darknet-19骨干网络*
- 19个卷积层
- 全局平均池化
- 1×1卷积降维

**YOLOv3**

*多尺度预测*
- 3个不同尺度的特征图
- 每个尺度3个锚点
- 总共9个锚点

*Darknet-53*
- 53个卷积层
- 残差连接
- 无池化层（步长卷积下采样）

*逻辑回归*
- 多标签分类
- 独立的逻辑分类器
- 处理重叠类别

**SSD（Single Shot MultiBox Detector）**

*多尺度特征图*
```
VGG-16基础网络 + 额外卷积层
特征图尺寸: 38×38, 19×19, 10×10, 5×5, 3×3, 1×1
```

*默认框*
- 每个特征图位置k个默认框
- 不同尺度和长宽比
- 总共8732个默认框

*损失函数*
```
L = (1/N)(L_conf + αL_loc)
L_conf: 置信度损失（交叉熵）
L_loc: 位置损失（Smooth L1）
```

*数据增强*
- 随机采样
- 水平翻转
- 颜色扭曲
- 随机扩展

#### 现代检测器

**FPN（Feature Pyramid Networks）**

*特征金字塔*
```
自顶向下路径 + 横向连接
P5 ← C5
P4 ← C4 + 上采样(P5)
P3 ← C3 + 上采样(P4)
P2 ← C2 + 上采样(P3)
```

*优势*
- 多尺度特征融合
- 语义信息丰富
- 小目标检测改善

**RetinaNet**

*焦点损失（Focal Loss）*
```
FL(pₜ) = -αₜ(1-pₜ)^γ log(pₜ)
```
- 解决类别不平衡问题
- 关注困难样本
- γ：聚焦参数（通常为2）
- αₜ：平衡参数

*网络结构*
- ResNet + FPN骨干
- 分类和回归子网络
- 锚点机制

**FCOS（Fully Convolutional One-Stage）**

*无锚点检测*
- 直接预测目标边界
- 避免锚点超参数
- 简化网络设计

*预测目标*
```
每个位置(x,y)预测:
- 分类分数: C维向量
- 边界距离: (l,t,r,b)
- 中心度: centerness
```

*中心度*
```
centerness = √(min(l,r)/max(l,r) × min(t,b)/max(t,b))
```

### 2.2 实例分割

#### Mask R-CNN

**架构扩展**
```
Faster R-CNN + Mask分支
输入: ROI特征
输出: K个m×m掩码（K个类别）
```

**ROIAlign**
- 解决ROI池化的量化误差
- 双线性插值
- 像素级对齐

**多任务损失**
```
L = L_cls + L_box + L_mask
L_mask = -(1/m²)∑∑[y_ij log ŷ_ij^k + (1-y_ij)log(1-ŷ_ij^k)]
```
- 每个类别独立的二值掩码
- 避免类间竞争

**训练策略**
- 正样本：IoU > 0.5
- 负样本：IoU < 0.5
- 掩码损失只在正样本上计算

#### 全景分割

**任务定义**
- 语义分割：像素级分类
- 实例分割：目标检测+掩码
- 全景分割：统一框架

**全景质量（PQ）**
```
PQ = (∑(p,g)∈TP IoU(p,g)) / (|TP| + (1/2)|FP| + (1/2)|FN|)
   = SQ × RQ
```
- SQ：分割质量
- RQ：识别质量
- 统一评估指标

**Panoptic FPN**
- 语义分割分支
- 实例分割分支
- 后处理融合

### 2.3 语义分割

#### 全卷积网络

**FCN（Fully Convolutional Networks）**

*核心思想*
- 全连接层 → 卷积层
- 任意输入尺寸
- 密集预测

*网络结构*
```
VGG-16骨干网络
FC6 → Conv6 (7×7, 4096)
FC7 → Conv7 (1×1, 4096)
FC8 → Conv8 (1×1, num_classes)
```

*上采样策略*
- FCN-32s：32倍上采样
- FCN-16s：融合pool4特征
- FCN-8s：融合pool3特征

*跳跃连接*
```
融合不同层次的特征
细节信息 + 语义信息
```

#### 编码器-解码器架构

**U-Net**

*网络结构*
```
编码器（收缩路径）:
卷积 → 卷积 → 最大池化 → 下采样

解码器（扩展路径）:
上采样 → 卷积 → 卷积 → 跳跃连接
```

*跳跃连接*
- 对应层特征拼接
- 保留空间信息
- 精确定位

*应用*
- 医学图像分割
- 生物图像分析
- 小数据集效果好

**SegNet**

*对称结构*
- VGG-16编码器
- 镜像解码器
- 池化索引传递

*上采样*
```
记录最大池化的索引位置
解码时使用索引进行上采样
减少参数量
```

#### 空洞卷积网络

**DeepLab系列**

*DeepLabv1*
- 空洞卷积扩大感受野
- 全连接CRF后处理
- 多尺度处理

*空洞卷积*
```
y[i] = ∑ x[i + r·k] w[k]
```
- r：空洞率（dilation rate）
- 保持分辨率
- 扩大感受野

*DeepLabv2*
- 空洞空间金字塔池化（ASPP）
- 多尺度上下文
- ResNet骨干网络

*ASPP模块*
```
并行空洞卷积:
rate=1: 1×1卷积
rate=6: 3×3空洞卷积
rate=12: 3×3空洞卷积
rate=18: 3×3空洞卷积
全局平均池化
```

*DeepLabv3*
- 改进ASPP
- 批归一化
- 级联空洞卷积

*DeepLabv3+*
- 编码器-解码器结构
- Xception骨干网络
- 深度可分离卷积

#### 注意力机制

**PSPNet（Pyramid Scene Parsing）**

*金字塔池化模块*
```
不同尺度的平均池化:
1×1, 2×2, 3×3, 6×6
上采样到原始大小
特征拼接
```

*全局上下文*
- 场景理解
- 多尺度信息
- 减少误分类

**Non-local Networks**

*非局部操作*
```
yᵢ = (1/C(x)) ∑ⱼ f(xᵢ,xⱼ)g(xⱼ)
```
- f(xᵢ,xⱼ)：相似度函数
- g(xⱼ)：表示函数
- C(x)：归一化因子

*自注意力*
- 长距离依赖
- 全局信息聚合
- 计算复杂度高

## 第三章：生成对抗网络

### 3.1 GAN基础理论

#### 博弈论框架

**极小极大博弈**
```
min_G max_D V(D,G) = E_x~p_data[log D(x)] + E_z~p_z[log(1-D(G(z)))]
```
- G：生成器，最小化目标函数
- D：判别器，最大化目标函数
- 零和博弈

**纳什均衡**
- 生成器和判别器都达到最优
- 理论上的收敛点
- 实际训练中难以达到

**全局最优解**
```
当 p_g = p_data 时:
D*(x) = 1/2
C(G) = -log(4)
```

#### 训练算法

**原始GAN算法**
```
for epoch in epochs:
    for k steps:
        # 训练判别器
        sample minibatch from data
        sample minibatch from noise
        update D by ascending gradient:
        ∇_θd (1/m)∑[log D(x^(i)) + log(1-D(G(z^(i))))]
    
    # 训练生成器
    sample minibatch from noise
    update G by descending gradient:
    ∇_θg (1/m)∑log(1-D(G(z^(i))))
```

**训练技巧**
- 判别器训练k步，生成器训练1步
- 使用-log D(G(z))代替log(1-D(G(z)))
- 批归一化
- LeakyReLU激活函数

#### 训练挑战

**模式崩溃（Mode Collapse）**
- 生成器只生成少数几种样本
- 缺乏多样性
- 判别器过强导致

**梯度消失**
- 判别器过于完美
- 生成器梯度接近零
- 训练停滞

**训练不稳定**
- 生成器和判别器不平衡
- 振荡而非收敛
- 超参数敏感

### 3.2 GAN变体

#### DCGAN

**深度卷积GAN**

*网络架构指导原则*
1. 用步长卷积替代池化
2. 在生成器和判别器中使用批归一化
3. 移除全连接层
4. 生成器使用ReLU，输出层使用Tanh
5. 判别器使用LeakyReLU

*生成器结构*
```
输入: 100维噪声向量
全连接: 100 → 4×4×1024
反卷积1: 4×4×1024 → 8×8×512
反卷积2: 8×8×512 → 16×16×256
反卷积3: 16×16×256 → 32×32×128
反卷积4: 32×32×128 → 64×64×3
```

*判别器结构*
```
输入: 64×64×3图像
卷积1: 64×64×3 → 32×32×128
卷积2: 32×32×128 → 16×16×256
卷积3: 16×16×256 → 8×8×512
卷积4: 8×8×512 → 4×4×1024
全连接: 4×4×1024 → 1
```

#### Conditional GAN

**条件生成**
```
min_G max_D V(D,G) = E_x~p_data[log D(x|y)] + E_z~p_z[log(1-D(G(z|y)))]
```
- y：条件信息（类别标签、文本等）
- 可控生成
- 多模态学习

**网络修改**
- 生成器：G(z,y)
- 判别器：D(x,y)
- 条件信息嵌入

**应用**
- 类别条件图像生成
- 文本到图像生成
- 图像到图像翻译

#### Wasserstein GAN

**Wasserstein距离**
```
W(P_r, P_g) = inf_{γ∈Π(P_r,P_g)} E_{(x,y)~γ}[||x-y||]
```
- 更稳定的距离度量
- 连续性保证
- 有意义的损失函数

**WGAN目标函数**
```
min_G max_{D∈1-Lipschitz} E_x~p_data[D(x)] - E_z~p_z[D(G(z))]
```
- 1-Lipschitz约束
- 权重裁剪实现

**WGAN-GP**
- 梯度惩罚代替权重裁剪
- 更稳定的训练
- 更好的收敛性

*梯度惩罚*
```
λE_x̂~p_x̂[(||∇_x̂ D(x̂)||_2 - 1)²]
其中 x̂ = εx + (1-ε)G(z), ε~U[0,1]
```

#### Progressive GAN

**渐进式训练**
- 从低分辨率开始训练
- 逐步增加分辨率
- 稳定训练过程

**网络增长**
```
4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128 → 256×256 → 512×512 → 1024×1024
```

**平滑过渡**
- 新层权重从0开始
- 线性插值过渡
- 避免训练震荡

**技术细节**
- 像素归一化
- 等化学习率
- 小批量标准差

### 3.3 应用与扩展

#### 图像到图像翻译

**Pix2Pix**

*条件GAN框架*
```
L_cGAN = E_x,y[log D(x,y)] + E_x,z[log(1-D(x,G(x,z)))]
L_L1 = E_x,y,z[||y - G(x,z)||_1]
G* = arg min_G max_D L_cGAN + λL_L1
```

*U-Net生成器*
- 编码器-解码器结构
- 跳跃连接
- 保留细节信息

*PatchGAN判别器*
- 局部判别
- N×N感受野
- 高频细节关注

**CycleGAN**

*循环一致性*
```
L_cyc = E_x~p_X[||F(G(x)) - x||_1] + E_y~p_Y[||G(F(y)) - y||_1]
```
- 无需配对数据
- 双向映射
- 循环一致性约束

*完整目标函数*
```
L = L_GAN(G,D_Y,X,Y) + L_GAN(F,D_X,Y,X) + λL_cyc(G,F)
```

**StarGAN**

*多域翻译*
- 单一生成器
- 多个域标签
- 统一框架

*域分类损失*
```
L_cls^r = E_x,c'[-log D_cls(c'|x)]
L_cls^f = E_x,c[-log D_cls(c|G(x,c))]
```

#### 超分辨率

**SRGAN**

*感知损失*
```
L_SR = L_MSE + αL_Gen + βL_VGG
```
- L_MSE：像素级损失
- L_Gen：对抗损失
- L_VGG：感知损失（VGG特征）

*VGG损失*
```
L_VGG = (1/W_i,j H_i,j) ∑∑(φ_i,j(I^HR) - φ_i,j(G(I^LR)))²
```
- φ_i,j：VGG网络第i层第j个特征图
- 感知相似性

**ESRGAN**

*改进*
- 残差密集块（RRDB）
- 相对判别器
- 感知损失改进
- 网络插值

#### 人脸生成与编辑

**StyleGAN**

*风格控制*
```
映射网络: z → w
合成网络: w → 图像
```

*AdaIN（自适应实例归一化）*
```
AdaIN(x_i, y) = y_s,i (x_i - μ(x_i))/σ(x_i) + y_b,i
```
- y_s,i, y_b,i：从w学习的风格参数
- 控制每层的风格

*风格混合*
- 不同层使用不同的w
- 粗糙到精细的控制
- 解耦表示

**StyleGAN2**

*改进*
- 权重去调制
- 路径长度正则化
- 无渐进式训练
- 更好的图像质量

## 实践项目

### 项目一：图像分类系统

**数据集准备**
```python
import torch
import torchvision
from torchvision import transforms

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
```

**ResNet实现**
```python
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
```

**训练循环**
```python
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {train_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.3f}%')

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f'Test Loss: {test_loss/len(testloader):.3f}, Test Acc: {100.*correct/total:.3f}%')
    return 100.*correct/total

# 训练
for epoch in range(200):
    train(epoch)
    acc = test()
    scheduler.step()
```

### 项目二：目标检测系统

**YOLO实现**
```python
class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.backbone = Darknet53()
        
        # 检测头
        self.conv_set_1 = self._make_conv_set(1024, 512)
        self.conv_1x1_1 = nn.Conv2d(512, 3 * (5 + num_classes), 1)
        
        self.conv_set_2 = self._make_conv_set(768, 256)
        self.conv_1x1_2 = nn.Conv2d(256, 3 * (5 + num_classes), 1)
        
        self.conv_set_3 = self._make_conv_set(384, 128)
        self.conv_1x1_3 = nn.Conv2d(128, 3 * (5 + num_classes), 1)
    
    def _make_conv_set(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch * 2, 3, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch * 2, 3, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        
        # 大目标检测 (13x13)
        p5 = self.conv_set_1(c5)
        pred_1 = self.conv_1x1_1(p5)
        
        # 中目标检测 (26x26)
        p5_up = F.interpolate(p5, scale_factor=2, mode='nearest')
        p4 = torch.cat([c4, p5_up], dim=1)
        p4 = self.conv_set_2(p4)
        pred_2 = self.conv_1x1_2(p4)
        
        # 小目标检测 (52x52)
        p4_up = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = torch.cat([c3, p4_up], dim=1)
        p3 = self.conv_set_3(p3)
        pred_3 = self.conv_1x1_3(p3)
        
        return pred_1, pred_2, pred_3
```

**损失函数**
```python
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        device = predictions[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        for i, pred in enumerate(predictions):
            b, a, gj, gi, tbox, tcls, tconf = self.build_targets(pred, targets, i)
            
            # 分类损失
            if tcls.shape[0]:
                lcls += self.bce_loss(pred[b, a, gj, gi, 5:], tcls)
            
            # 目标性损失
            lobj += self.bce_loss(pred[..., 4], tconf)
            
            # 边界框损失
            if tbox.shape[0]:
                pbox = pred[b, a, gj, gi, :4]
                lbox += (1.0 - bbox_iou(pbox, tbox)).mean()
        
        return lcls + lbox + lobj
```

### 项目三：语义分割

**U-Net实现**
```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits
```

### 项目四：GAN图像生成

**DCGAN实现**
```python
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入 (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
```

**训练循环**
```python
# 训练DCGAN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = Generator().to(device)
netD = Discriminator().to(device)

# 损失函数
criterion = nn.BCELoss()

# 优化器
optimD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # 更新判别器
        netD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1, device=device)
        
        output = netD(real_data)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        # 训练假数据
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimD.step()
        
        # 更新生成器
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimG.step()
```

## 学习评估

### 理论评估
1. **概念理解**：计算机视觉基础概念
2. **算法原理**：深度学习模型原理
3. **技术对比**：不同方法的优缺点
4. **应用分析**：实际问题解决方案

### 实践评估
1. **编程实现**：核心算法编程
2. **模型训练**：完整训练流程
3. **性能优化**：模型调优技巧
4. **项目应用**：实际项目开发

### 综合评估
1. **技术报告**：深度技术分析
2. **代码审查**：代码质量评估
3. **演示展示**：项目成果展示
4. **创新应用**：技术创新能力

## 延伸学习

### 前沿研究方向
1. **Vision Transformer**：视觉注意力机制
2. **自监督学习**：无标签数据利用
3. **神经架构搜索**：自动网络设计
4. **多模态学习**：视觉-语言融合
5. **3D视觉**：三维场景理解

### 应用领域
1. **自动驾驶**：环境感知与决策
2. **医学影像**：疾病诊断辅助
3. **工业检测**：质量控制自动化
4. **安防监控**：智能视频分析
5. **增强现实**：虚实融合技术

### 工具和框架
1. **深度学习框架**：PyTorch, TensorFlow
2. **计算机视觉库**：OpenCV, PIL
3. **数据处理**：NumPy, Pandas
4. **可视化工具**：Matplotlib, Tensorboard
5. **部署工具**：ONNX, TensorRT

## 总结

计算机视觉是人工智能的重要分支，从传统的图像处理方法到现代的深度学习技术，经历了巨大的发展。本模块系统介绍了计算机视觉的核心技术，包括图像分类、目标检测、语义分割和生成对抗网络等。

通过理论学习和实践项目，学生将掌握：
1. 计算机视觉的基础理论和核心概念
2. 深度学习在视觉任务中的应用
3. 主流模型的设计原理和实现方法
4. 实际项目的开发和部署技能

这些知识和技能将为学生在计算机视觉领域的进一步研究和应用奠定坚实基础。