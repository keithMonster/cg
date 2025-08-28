# 模块九：生成式AI与创意计算

## 学习目标

通过本模块的学习，学生将能够：

1. **理解生成式AI的基本原理**：掌握生成模型的数学基础和核心概念
2. **掌握主流生成模型**：深入理解VAE、GAN、扩散模型等关键技术
3. **实现创意计算系统**：构建能够进行艺术创作、内容生成的AI系统
4. **探索多模态生成**：学习文本、图像、音频等多模态内容生成技术
5. **应用生成式AI**：在实际项目中运用生成式AI解决创意和内容生成问题

## 第一章：生成式AI基础理论

### 1.1 生成模型概述

#### 什么是生成式AI？

**简单理解**

生成式AI就像是一个"超级创作家"，它能够：
- **看懂规律**：通过学习大量例子，理解事物的规律和特征
- **创造新内容**：根据学到的规律，创造出全新的、从未见过的内容
- **模仿风格**：能够模仿特定的风格或特征来创作

**生活中的例子**：
- **画家AI**：学习了毕加索的画风后，能画出新的"毕加索风格"作品
- **作家AI**：读了大量小说后，能写出新的故事
- **音乐家AI**：听了很多古典音乐后，能作出新的古典乐曲
- **设计师AI**：看了很多Logo设计后，能设计出新的Logo

#### 生成模型的工作原理

**三步走策略**：

1. **学习阶段 - 像学生做作业**
   - 给AI看大量的例子（比如10万张猫的照片）
   - AI分析这些例子，找出共同特征（猫有尖耳朵、胡须、四条腿等）
   - 建立一个"猫的概念模型"

2. **理解阶段 - 像总结规律**
   - AI把学到的特征整理成"创作规则"
   - 比如："猫的耳朵通常是三角形的"、"猫的眼睛通常是椭圆形的"
   - 这些规则存储在AI的"大脑"里

3. **创作阶段 - 像艺术家创作**
   - 根据学到的规则，AI开始创作新内容
   - 比如画一只从未存在过的猫，但看起来很像真猫
   - 每次创作都是独一无二的

**概率生成模型的核心思想**

想象你要教一个从未见过猫的外星人画猫：
- 你会告诉它："90%的猫有尖耳朵，80%的猫有长尾巴，70%的猫有条纹"
- 外星人根据这些概率信息，就能画出看起来像猫的图
- 这就是概率生成模型的基本思路：用概率来描述事物的特征
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class GenerativeModel(nn.Module):
    """生成模型基类"""
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
    
    def sample_prior(self, batch_size):
        """从先验分布采样"""
        return torch.randn(batch_size, self.latent_dim)
    
    def generate(self, z):
        """从潜在变量生成数据"""
        raise NotImplementedError
    
    def log_likelihood(self, x):
        """计算数据的对数似然"""
        raise NotImplementedError
    
    def sample(self, num_samples):
        """生成样本"""
        z = self.sample_prior(num_samples)
        return self.generate(z)

class SimpleGaussianMixture(GenerativeModel):
    """简单高斯混合模型"""
    def __init__(self, num_components, data_dim):
        super().__init__(num_components, data_dim)
        self.num_components = num_components
        
        # 混合权重
        self.weights = nn.Parameter(torch.ones(num_components))
        # 均值
        self.means = nn.Parameter(torch.randn(num_components, data_dim))
        # 协方差（简化为对角矩阵）
        self.log_stds = nn.Parameter(torch.zeros(num_components, data_dim))
    
    def forward(self, x):
        """计算对数似然"""
        batch_size = x.size(0)
        
        # 计算每个组件的概率密度
        log_probs = []
        for k in range(self.num_components):
            dist = Normal(self.means[k], torch.exp(self.log_stds[k]))
            log_prob = dist.log_prob(x).sum(dim=1)  # 假设特征独立
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, num_components]
        
        # 加权求和
        weights = F.softmax(self.weights, dim=0)
        log_weights = torch.log(weights).unsqueeze(0).expand(batch_size, -1)
        
        # 使用logsumexp技巧计算混合概率
        log_likelihood = torch.logsumexp(log_probs + log_weights, dim=1)
        
        return log_likelihood
    
    def sample(self, num_samples):
        """从混合模型采样"""
        # 选择组件
        weights = F.softmax(self.weights, dim=0)
        component_dist = Categorical(weights)
        components = component_dist.sample((num_samples,))
        
        # 从选定组件采样
        samples = []
        for i in range(num_samples):
            k = components[i]
            dist = Normal(self.means[k], torch.exp(self.log_stds[k]))
            sample = dist.sample()
            samples.append(sample)
        
        return torch.stack(samples)
```

#### 信息论基础

**熵与互信息**
```python
class InformationTheory:
    """信息论工具类"""
    
    @staticmethod
    def entropy(p):
        """计算熵"""
        # p: 概率分布 [batch_size, num_classes]
        log_p = torch.log(p + 1e-8)  # 避免log(0)
        entropy = -torch.sum(p * log_p, dim=1)
        return entropy
    
    @staticmethod
    def kl_divergence(p, q):
        """计算KL散度 KL(p||q)"""
        log_p = torch.log(p + 1e-8)
        log_q = torch.log(q + 1e-8)
        kl = torch.sum(p * (log_p - log_q), dim=1)
        return kl
    
    @staticmethod
    def mutual_information(x, y, bins=50):
        """估计互信息"""
        # 简化的互信息估计（基于直方图）
        # 实际应用中可能需要更复杂的估计方法
        
        # 转换为numpy进行直方图计算
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # 计算联合分布和边际分布
        joint_hist, x_edges, y_edges = np.histogram2d(x_np, y_np, bins=bins)
        joint_prob = joint_hist / np.sum(joint_hist)
        
        x_prob = np.sum(joint_prob, axis=1)
        y_prob = np.sum(joint_prob, axis=0)
        
        # 计算互信息
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (x_prob[i] * y_prob[j] + 1e-8)
                    )
        
        return mi
    
    @staticmethod
    def js_divergence(p, q):
        """计算JS散度"""
        m = 0.5 * (p + q)
        js = 0.5 * InformationTheory.kl_divergence(p, m) + \
             0.5 * InformationTheory.kl_divergence(q, m)
        return js

# 使用示例
info_theory = InformationTheory()

# 创建两个概率分布
p = torch.softmax(torch.randn(100, 10), dim=1)
q = torch.softmax(torch.randn(100, 10), dim=1)

# 计算各种信息论量
entropy_p = info_theory.entropy(p)
kl_pq = info_theory.kl_divergence(p, q)
js_pq = info_theory.js_divergence(p, q)

print(f"熵: {entropy_p.mean():.4f}")
print(f"KL散度: {kl_pq.mean():.4f}")
print(f"JS散度: {js_pq.mean():.4f}")
```

### 1.2 变分推理

#### 变分自编码器（VAE）

**VAE实现**
```python
class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 潜在变量参数
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 假设输入在[0,1]范围
        )
    
    def encode(self, x):
        """编码"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE损失函数"""
        # 重构损失
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # β-VAE
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device='cpu'):
        """生成样本"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
        return samples

# 训练VAE
def train_vae(model, dataloader, optimizer, device, beta=1.0):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        recon_batch, mu, logvar = model(data)
        
        # 计算损失
        loss, recon_loss, kl_loss = model.loss_function(
            recon_batch, data, mu, logvar, beta
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')
    
    return total_loss / len(dataloader.dataset)
```

#### β-VAE与解耦表示

**解耦度量**
```python
class DisentanglementMetrics:
    """解耦度量"""
    
    @staticmethod
    def beta_vae_metric(model, dataset, num_samples=1000):
        """β-VAE解耦度量"""
        model.eval()
        
        # 生成数据
        with torch.no_grad():
            # 固定其他维度，只变化一个维度
            base_z = torch.randn(1, model.latent_dim)
            
            disentanglement_scores = []
            
            for dim in range(model.latent_dim):
                # 在第dim维度上变化
                z_varied = base_z.repeat(num_samples, 1)
                z_varied[:, dim] = torch.linspace(-3, 3, num_samples)
                
                # 生成图像
                generated = model.decode(z_varied)
                
                # 计算变化程度（简化度量）
                variance = torch.var(generated, dim=0).mean()
                disentanglement_scores.append(variance.item())
        
        return disentanglement_scores
    
    @staticmethod
    def mig_score(model, dataset, num_factors=10):
        """互信息间隙（MIG）评分"""
        # 这是一个简化版本，实际实现需要更复杂的因子识别
        model.eval()
        
        with torch.no_grad():
            # 采样潜在变量和生成数据
            z_samples = torch.randn(1000, model.latent_dim)
            x_samples = model.decode(z_samples)
            
            # 计算每个潜在维度与生成因子的互信息
            mi_matrix = torch.zeros(model.latent_dim, num_factors)
            
            for i in range(model.latent_dim):
                for j in range(num_factors):
                    # 简化的互信息计算
                    mi = InformationTheory.mutual_information(
                        z_samples[:, i], x_samples[:, j]
                    )
                    mi_matrix[i, j] = mi
            
            # 计算MIG分数
            # MIG = (最大MI - 第二大MI) / 熵
            sorted_mi, _ = torch.sort(mi_matrix, dim=1, descending=True)
            mig_scores = (sorted_mi[:, 0] - sorted_mi[:, 1]) / torch.sum(sorted_mi, dim=1)
            
        return mig_scores.mean().item()
```

## 第二章：生成对抗网络（GAN）

### 2.1 GAN基础理论

#### 原始GAN

**GAN实现**
```python
class Generator(nn.Module):
    """生成器"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """判别器"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class GAN:
    """GAN训练器"""
    def __init__(self, generator, discriminator, latent_dim, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device = device
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_data):
        """训练一步"""
        batch_size = real_data.size(0)
        
        # 真实和虚假标签
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # ==================== 训练判别器 ====================
        self.d_optimizer.zero_grad()
        
        # 真实数据
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # 生成假数据
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z).detach()  # 不计算生成器梯度
        fake_output = self.discriminator(fake_data)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # ==================== 训练生成器 ====================
        self.g_optimizer.zero_grad()
        
        # 生成假数据（这次需要梯度）
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data)
        
        # 生成器损失（希望判别器认为生成的数据是真的）
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_samples(self, num_samples):
        """生成样本"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
        self.generator.train()
        return samples
```

### 2.2 GAN变体

#### DCGAN（深度卷积GAN）

**DCGAN实现**
```python
class DCGANGenerator(nn.Module):
    """DCGAN生成器"""
    def __init__(self, latent_dim, num_channels=3, feature_map_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # 状态: (feature_map_size*8) x 4 x 4
            
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # 状态: (feature_map_size*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # 状态: (feature_map_size*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # 状态: (feature_map_size) x 32 x 32
            
            nn.ConvTranspose2d(feature_map_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: num_channels x 64 x 64
        )
    
    def forward(self, z):
        # 重塑输入为4D张量
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)

class DCGANDiscriminator(nn.Module):
    """DCGAN判别器"""
    def __init__(self, num_channels=3, feature_map_size=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 输入: num_channels x 64 x 64
            nn.Conv2d(num_channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: feature_map_size x 32 x 32
            
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_map_size*2) x 16 x 16
            
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_map_size*4) x 8 x 8
            
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_map_size*8) x 4 x 4
            
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: 1 x 1 x 1
        )
    
    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)

# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

#### Wasserstein GAN

**WGAN实现**
```python
class WGANCritic(nn.Module):
    """WGAN批评器（不使用sigmoid）"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
            # 注意：没有sigmoid激活
        )
    
    def forward(self, x):
        return self.model(x)

class WGAN:
    """Wasserstein GAN"""
    def __init__(self, generator, critic, latent_dim, device, clip_value=0.01):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.latent_dim = latent_dim
        self.device = device
        self.clip_value = clip_value
        
        # 优化器（WGAN推荐使用RMSprop）
        self.g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
        self.c_optimizer = torch.optim.RMSprop(critic.parameters(), lr=5e-5)
    
    def train_step(self, real_data, n_critic=5):
        """训练一步"""
        batch_size = real_data.size(0)
        
        # ==================== 训练批评器 ====================
        for _ in range(n_critic):
            self.c_optimizer.zero_grad()
            
            # 真实数据
            real_output = self.critic(real_data)
            
            # 生成假数据
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_data = self.generator(z).detach()
            fake_output = self.critic(fake_data)
            
            # Wasserstein损失
            c_loss = -torch.mean(real_output) + torch.mean(fake_output)
            c_loss.backward()
            self.c_optimizer.step()
            
            # 权重裁剪
            for p in self.critic.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)
        
        # ==================== 训练生成器 ====================
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        fake_output = self.critic(fake_data)
        
        # 生成器损失
        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        self.g_optimizer.step()
        
        return c_loss.item(), g_loss.item()

# WGAN-GP（梯度惩罚）
class WGAN_GP(WGAN):
    """带梯度惩罚的WGAN"""
    def __init__(self, generator, critic, latent_dim, device, lambda_gp=10):
        super().__init__(generator, critic, latent_dim, device)
        self.lambda_gp = lambda_gp
        
        # 使用Adam优化器
        self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.c_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    def gradient_penalty(self, real_data, fake_data):
        """计算梯度惩罚"""
        batch_size = real_data.size(0)
        
        # 随机插值
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 计算插值点的批评器输出
        interpolated_output = self.critic(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=interpolated_output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_output),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def train_step(self, real_data, n_critic=5):
        """训练一步（带梯度惩罚）"""
        batch_size = real_data.size(0)
        
        # ==================== 训练批评器 ====================
        for _ in range(n_critic):
            self.c_optimizer.zero_grad()
            
            # 真实数据
            real_output = self.critic(real_data)
            
            # 生成假数据
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_data = self.generator(z).detach()
            fake_output = self.critic(fake_data)
            
            # Wasserstein损失
            wasserstein_loss = -torch.mean(real_output) + torch.mean(fake_output)
            
            # 梯度惩罚
            gp = self.gradient_penalty(real_data, fake_data)
            
            # 总损失
            c_loss = wasserstein_loss + self.lambda_gp * gp
            c_loss.backward()
            self.c_optimizer.step()
        
        # ==================== 训练生成器 ====================
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z)
        fake_output = self.critic(fake_data)
        
        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        self.g_optimizer.step()
        
        return c_loss.item(), g_loss.item()
```

### 2.3 条件生成

#### 条件GAN（cGAN）

**cGAN实现**
```python
class ConditionalGenerator(nn.Module):
    """条件生成器"""
    def __init__(self, latent_dim, num_classes, hidden_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # 生成器网络
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),  # 噪声 + 标签嵌入
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # 获取标签嵌入
        label_emb = self.label_embedding(labels)
        
        # 连接噪声和标签
        input_tensor = torch.cat([z, label_emb], dim=1)
        
        return self.model(input_tensor)

class ConditionalDiscriminator(nn.Module):
    """条件判别器"""
    def __init__(self, input_dim, num_classes, hidden_dim):
        super().__init__()
        self.num_classes = num_classes
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, input_dim)
        
        # 判别器网络
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 数据 + 标签嵌入
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # 获取标签嵌入
        label_emb = self.label_embedding(labels)
        
        # 连接数据和标签
        input_tensor = torch.cat([x, label_emb], dim=1)
        
        return self.model(input_tensor)

class ConditionalGAN:
    """条件GAN训练器"""
    def __init__(self, generator, discriminator, latent_dim, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device = device
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_data, real_labels):
        """训练一步"""
        batch_size = real_data.size(0)
        
        # 标签
        real_labels_tensor = torch.ones(batch_size, 1).to(self.device)
        fake_labels_tensor = torch.zeros(batch_size, 1).to(self.device)
        
        # ==================== 训练判别器 ====================
        self.d_optimizer.zero_grad()
        
        # 真实数据
        real_output = self.discriminator(real_data, real_labels)
        d_loss_real = self.criterion(real_output, real_labels_tensor)
        
        # 生成假数据
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z, real_labels).detach()
        fake_output = self.discriminator(fake_data, real_labels)
        d_loss_fake = self.criterion(fake_output, fake_labels_tensor)
        
        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # ==================== 训练生成器 ====================
        self.g_optimizer.zero_grad()
        
        # 生成假数据
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z, real_labels)
        fake_output = self.discriminator(fake_data, real_labels)
        
        # 生成器损失
        g_loss = self.criterion(fake_output, real_labels_tensor)
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_samples(self, labels, num_samples=None):
        """生成指定类别的样本"""
        if num_samples is None:
            num_samples = len(labels)
        
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z, labels)
        self.generator.train()
        return samples
```

## 第三章：扩散模型

### 3.1 扩散过程理论

#### 前向扩散过程

**扩散模型基础**
```python
import math

class DiffusionSchedule:
    """扩散调度器"""
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 线性调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 用于采样的预计算值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于逆向过程的预计算值
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差"""
        posterior_mean = (
            self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

class UNet(nn.Module):
    """U-Net噪声预测网络"""
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128, base_channels=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # 下采样路径
        self.down1 = self._make_layer(in_channels, base_channels, time_emb_dim * 4)
        self.down2 = self._make_layer(base_channels, base_channels * 2, time_emb_dim * 4)
        self.down3 = self._make_layer(base_channels * 2, base_channels * 4, time_emb_dim * 4)
        
        # 瓶颈层
        self.bottleneck = self._make_layer(base_channels * 4, base_channels * 8, time_emb_dim * 4)
        
        # 上采样路径
        self.up3 = self._make_layer(base_channels * 8 + base_channels * 4, base_channels * 4, time_emb_dim * 4)
        self.up2 = self._make_layer(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim * 4)
        self.up1 = self._make_layer(base_channels * 2 + base_channels, base_channels, time_emb_dim * 4)
        
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _make_layer(self, in_channels, out_channels, time_emb_dim):
        """创建包含时间嵌入的卷积层"""
        return ResBlock(in_channels, out_channels, time_emb_dim)
    
    def positional_encoding(self, timesteps, dim):
        """位置编码用于时间嵌入"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
    
    def forward(self, x, timesteps):
        # 时间嵌入
        t_emb = self.positional_encoding(timesteps, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # 下采样
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        x3 = self.down3(self.pool(x2), t_emb)
        
        # 瓶颈
        x_bottleneck = self.bottleneck(self.pool(x3), t_emb)
        
        # 上采样
        x = self.up3(torch.cat([self.upsample(x_bottleneck), x3], dim=1), t_emb)
        x = self.up2(torch.cat([self.upsample(x), x2], dim=1), t_emb)
        x = self.up1(torch.cat([self.upsample(x), x1], dim=1), t_emb)
        
        # 输出
        return self.out_conv(x)

class ResBlock(nn.Module):
    """残差块（包含时间嵌入）"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.SiLU()
        
        # 跳跃连接
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # 添加时间嵌入
        time_proj = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_proj
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        # 跳跃连接
        return self.activation(h + self.skip_conv(x))
```

### 3.2 DDPM实现

**去噪扩散概率模型**
```python
class DDPM:
    """去噪扩散概率模型"""
    def __init__(self, model, schedule, device):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device
        
        # 将调度器参数移到设备
        for attr_name in dir(schedule):
            attr = getattr(schedule, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(schedule, attr_name, attr.to(device))
    
    def training_loss(self, x_start):
        """计算训练损失"""
        batch_size = x_start.size(0)
        
        # 随机选择时间步
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,), device=self.device)
        
        # 生成噪声
        noise = torch.randn_like(x_start)
        
        # 前向扩散
        x_noisy = self.schedule.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_noisy, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def p_sample(self, x, t, t_index):
        """单步去噪采样"""
        betas_t = self.schedule.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.schedule.alphas[t]).reshape(-1, 1, 1, 1)
        
        # 预测噪声
        predicted_noise = self.model(x, t)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.schedule.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, shape):
        """完整的采样循环"""
        device = next(self.model.parameters()).device
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.schedule.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        
        return img
    
    def sample(self, batch_size, image_size, channels=3):
        """生成样本"""
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

# 训练DDPM
def train_ddpm(model, dataloader, optimizer, device, num_epochs):
    ddpm = DDPM(model, DiffusionSchedule(), device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            optimizer.zero_grad()
            loss = ddpm.training_loss(data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        
        # 生成样本
        if epoch % 10 == 0:
            with torch.no_grad():
                samples = ddpm.sample(16, 32)
                # 保存或显示样本
                print(f'Generated samples shape: {samples.shape}')
```

### 3.3 DDIM与快速采样

**去噪扩散隐式模型**
```python
class DDIM:
    """去噪扩散隐式模型"""
    def __init__(self, model, schedule, device):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device
    
    def ddim_sample(self, x, t, t_prev, eta=0.0):
        """DDIM采样步骤"""
        # 预测噪声
        predicted_noise = self.model(x, t)
        
        # 获取alpha值
        alpha_prod_t = self.schedule.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        alpha_prod_t_prev = self.schedule.alphas_cumprod[t_prev].reshape(-1, 1, 1, 1)
        
        # 预测x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * predicted_noise) / torch.sqrt(alpha_prod_t)
        
        # 计算方向向量
        direction = torch.sqrt(1 - alpha_prod_t_prev - eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) * predicted_noise
        
        # 随机性项
        noise = torch.randn_like(x) if eta > 0 else 0
        
        # DDIM更新
        x_prev = torch.sqrt(alpha_prod_t_prev) * pred_x0 + direction + eta * torch.sqrt((1 - alpha_prod_t_prev) - direction**2 / (1 - alpha_prod_t_prev)) * noise
        
        return x_prev
    
    def ddim_sample_loop(self, shape, num_inference_steps=50, eta=0.0):
        """DDIM采样循环"""
        device = next(self.model.parameters()).device
        
        # 创建采样时间步
        timesteps = torch.linspace(self.schedule.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
            
            img = self.ddim_sample(img, t_batch, t_prev_batch, eta)
        
        return img
    
    def fast_sample(self, batch_size, image_size, channels=3, num_steps=20):
        """快速采样"""
        return self.ddim_sample_loop((batch_size, channels, image_size, image_size), num_steps)
```

## 第四章：多模态生成

### 4.1 文本到图像生成

#### CLIP引导生成

**CLIP引导的扩散模型**
```python
import clip

class CLIPGuidedDiffusion:
    """CLIP引导的扩散模型"""
    def __init__(self, diffusion_model, clip_model, clip_preprocess, device):
        self.diffusion_model = diffusion_model
        self.clip_model = clip_model.to(device)
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def clip_loss(self, image, text_prompt):
        """计算CLIP损失"""
        # 预处理图像
        if image.dim() == 4:  # batch
            processed_images = []
            for img in image:
                # 将图像从[-1,1]转换到[0,1]
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                
                # 调整大小并标准化
                img_pil = transforms.ToPILImage()(img.cpu())
                img_processed = self.clip_preprocess(img_pil).unsqueeze(0)
                processed_images.append(img_processed)
            
            processed_images = torch.cat(processed_images, dim=0).to(self.device)
        
        # 编码文本
        text_tokens = clip.tokenize([text_prompt] * image.size(0)).to(self.device)
        
        # 获取特征
        image_features = self.clip_model.encode_image(processed_images)
        text_features = self.clip_model.encode_text(text_tokens)
        
        # 计算相似度
        similarity = torch.cosine_similarity(image_features, text_features, dim=1)
        
        # 返回负相似度作为损失（我们想要最大化相似度）
        return -similarity.mean()
    
    def guided_sample_step(self, x, t, text_prompt, guidance_scale=7.5):
        """引导采样步骤"""
        x.requires_grad_(True)
        
        # 无条件预测
        with torch.no_grad():
            uncond_noise = self.diffusion_model.model(x, t)
        
        # 条件预测（通过梯度）
        # 这里我们使用CLIP损失的梯度来引导生成
        
        # 预测当前的x_0
        alpha_prod_t = self.diffusion_model.schedule.alphas_cumprod[t[0]]
        pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * uncond_noise) / torch.sqrt(alpha_prod_t)
        
        # 计算CLIP损失
        clip_loss = self.clip_loss(pred_x0, text_prompt)
        
        # 计算梯度
        grad = torch.autograd.grad(clip_loss, x)[0]
        
        # 引导的噪声预测
        guided_noise = uncond_noise - guidance_scale * grad
        
        x.requires_grad_(False)
        
        return guided_noise
    
    def text_to_image(self, text_prompt, image_size=256, num_steps=50, guidance_scale=7.5):
        """文本到图像生成"""
        batch_size = 1
        shape = (batch_size, 3, image_size, image_size)
        
        # 创建采样时间步
        timesteps = torch.linspace(
            self.diffusion_model.schedule.num_timesteps - 1, 0, num_steps, dtype=torch.long
        )
        
        # 从纯噪声开始
        img = torch.randn(shape, device=self.device)
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # 引导采样
            predicted_noise = self.guided_sample_step(img, t_batch, text_prompt, guidance_scale)
            
            # 更新图像
            alpha_prod_t = self.diffusion_model.schedule.alphas_cumprod[t]
            alpha_prod_t_prev = self.diffusion_model.schedule.alphas_cumprod[timesteps[i + 1]]
            
            pred_x0 = (img - torch.sqrt(1 - alpha_prod_t) * predicted_noise) / torch.sqrt(alpha_prod_t)
            direction = torch.sqrt(1 - alpha_prod_t_prev) * predicted_noise
            img = torch.sqrt(alpha_prod_t_prev) * pred_x0 + direction
        
        return img

# 使用示例
def setup_clip_guided_generation():
    """设置CLIP引导生成"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载CLIP模型
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # 创建扩散模型
    unet = UNet()
    schedule = DiffusionSchedule()
    diffusion_model = DDPM(unet, schedule, device)
    
    # 创建CLIP引导生成器
    clip_guided = CLIPGuidedDiffusion(diffusion_model, clip_model, clip_preprocess, device)
    
    return clip_guided

# 生成示例
clip_guided = setup_clip_guided_generation()
generated_image = clip_guided.text_to_image("a beautiful sunset over mountains")
```

### 4.2 音频生成

#### WaveNet架构

**WaveNet实现**
```python
class WaveNet(nn.Module):
    """WaveNet音频生成模型"""
    def __init__(self, num_layers=30, num_blocks=3, residual_channels=32, 
                 gate_channels=32, skip_channels=256, kernel_size=2, 
                 num_classes=256):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_classes = num_classes
        
        # 输入卷积
        self.start_conv = nn.Conv1d(1, residual_channels, 1)
        
        # 扩张卷积层
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for block in range(num_blocks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                
                # 扩张卷积（门控）
                dilated_conv = nn.Conv1d(
                    residual_channels, gate_channels * 2, kernel_size,
                    dilation=dilation, padding=dilation
                )
                self.dilated_convs.append(dilated_conv)
                
                # 残差连接
                residual_conv = nn.Conv1d(gate_channels, residual_channels, 1)
                self.residual_convs.append(residual_conv)
                
                # 跳跃连接
                skip_conv = nn.Conv1d(gate_channels, skip_channels, 1)
                self.skip_convs.append(skip_conv)
        
        # 输出层
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.end_conv2 = nn.Conv1d(skip_channels, num_classes, 1)
    
    def forward(self, x):
        # 输入处理
        x = self.start_conv(x)
        skip_connections = []
        
        # 扩张卷积块
        for i in range(len(self.dilated_convs)):
            # 门控激活
            gated = self.dilated_convs[i](x)
            filter_gate, gate_gate = gated.chunk(2, dim=1)
            gated = torch.tanh(filter_gate) * torch.sigmoid(gate_gate)
            
            # 残差连接
            residual = self.residual_convs[i](gated)
            x = x + residual
            
            # 跳跃连接
            skip = self.skip_convs[i](gated)
            skip_connections.append(skip)
        
        # 合并跳跃连接
        skip_sum = sum(skip_connections)
        
        # 输出层
        out = F.relu(skip_sum)
        out = F.relu(self.end_conv1(out))
        out = self.end_conv2(out)
        
        return out
    
    def generate(self, length, temperature=1.0):
        """生成音频序列"""
        self.eval()
        
        # 初始化
        generated = torch.zeros(1, 1, length)
        
        with torch.no_grad():
            for i in range(length):
                # 获取当前上下文
                context_length = min(i + 1, self.receptive_field())
                context = generated[:, :, max(0, i + 1 - context_length):i + 1]
                
                # 预测下一个样本
                logits = self.forward(context)
                probs = F.softmax(logits[:, :, -1] / temperature, dim=1)
                
                # 采样
                next_sample = torch.multinomial(probs, 1)
                
                # 转换为音频值
                audio_value = (next_sample.float() / (self.num_classes - 1)) * 2 - 1
                
                if i < length - 1:
                    generated[:, :, i + 1] = audio_value
        
        return generated
    
    def receptive_field(self):
        """计算感受野大小"""
        return (2 ** self.num_layers - 1) * self.num_blocks + 1

# 音频预处理
class AudioProcessor:
    """音频处理工具"""
    def __init__(self, sample_rate=16000, num_classes=256):
        self.sample_rate = sample_rate
        self.num_classes = num_classes
    
    def mu_law_encode(self, audio, mu=255):
        """μ-law编码"""
        mu = torch.tensor(mu, dtype=audio.dtype)
        safe_audio = torch.clamp(audio, -1, 1)
        magnitude = torch.log1p(mu * torch.abs(safe_audio)) / torch.log1p(mu)
        signal = torch.sign(safe_audio) * magnitude
        return ((signal + 1) / 2 * mu + 0.5).long()
    
    def mu_law_decode(self, encoded, mu=255):
        """μ-law解码"""
        mu = torch.tensor(mu, dtype=torch.float32)
        signal = 2 * (encoded.float() / mu) - 1
        magnitude = (1 / mu) * ((1 + mu) ** torch.abs(signal) - 1)
        return torch.sign(signal) * magnitude
    
    def preprocess_audio(self, audio_path):
        """预处理音频文件"""
        import librosa
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio = torch.from_numpy(audio).float()
        
        # 标准化到[-1, 1]
        audio = audio / torch.max(torch.abs(audio))
        
        # μ-law编码
        encoded = self.mu_law_encode(audio)
        
        return encoded

# 训练WaveNet
def train_wavenet(model, dataloader, optimizer, device, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, audio_batch in enumerate(dataloader):
            audio_batch = audio_batch.to(device)
            
            # 输入和目标
            input_audio = audio_batch[:, :, :-1]
            target_audio = audio_batch[:, 0, 1:]  # 移除通道维度
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(input_audio)
            
            # 计算损失
            loss = criterion(output.transpose(1, 2), target_audio.long())
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
```

### 4.3 视频生成

#### 3D卷积生成器

**视频生成模型**
```python
class Video3DGAN(nn.Module):
    """3D卷积视频生成器"""
    def __init__(self, latent_dim=100, num_frames=16, frame_size=64, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_channels = num_channels
        
        # 计算初始特征图大小
        self.init_size = frame_size // 8  # 经过3次上采样
        
        # 全连接层
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size * (num_frames // 8))
        
        # 3D转置卷积层
        self.conv_blocks = nn.Sequential(
            # 第一个块: (512, T/8, H/8, W/8) -> (256, T/4, H/4, W/4)
            nn.ConvTranspose3d(512, 256, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # 第二个块: (256, T/4, H/4, W/4) -> (128, T/2, H/2, W/2)
            nn.ConvTranspose3d(256, 128, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # 第三个块: (128, T/2, H/2, W/2) -> (64, T, H, W)
            nn.ConvTranspose3d(128, 64, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # 输出层: (64, T, H, W) -> (3, T, H, W)
            nn.ConvTranspose3d(64, num_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Tanh()
        )
    
    def forward(self, z):
        # 全连接层
        out = self.fc(z)
        
        # 重塑为5D张量 (batch, channels, depth, height, width)
        out = out.view(out.size(0), 512, self.num_frames // 8, self.init_size, self.init_size)
        
        # 3D卷积
        video = self.conv_blocks(out)
        
        return video

class Video3DDiscriminator(nn.Module):
    """3D卷积视频判别器"""
    def __init__(self, num_frames=16, frame_size=64, num_channels=3):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # 输入: (3, T, H, W)
            nn.Conv3d(num_channels, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, T, H/2, W/2)
            nn.Conv3d(64, 128, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, T/2, H/4, W/4)
            nn.Conv3d(128, 256, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (256, T/4, H/8, W/8)
            nn.Conv3d(256, 512, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, video):
        features = self.conv_blocks(video)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        output = self.classifier(flattened)
        return output

# 视频预处理
class VideoProcessor:
    """视频处理工具"""
    def __init__(self, frame_size=64, num_frames=16):
        self.frame_size = frame_size
        self.num_frames = num_frames
    
    def load_video(self, video_path):
        """加载视频文件"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 调整大小
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            # BGR转RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 标准化到[-1, 1]
            frame = (frame / 127.5) - 1.0
            
            frames.append(frame)
        
        cap.release()
        
        # 如果帧数不足，重复最后一帧
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        
        # 转换为张量 (T, H, W, C) -> (C, T, H, W)
        video_tensor = torch.from_numpy(np.array(frames)).float()
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        
        return video_tensor
    
    def save_video(self, video_tensor, output_path, fps=30):
        """保存视频张量为文件"""
        import cv2
        
        # (C, T, H, W) -> (T, H, W, C)
        video = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
        
        # 反标准化到[0, 255]
        video = ((video + 1.0) * 127.5).astype(np.uint8)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.frame_size, self.frame_size))
        
        for frame in video:
            # RGB转BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()

# 训练视频GAN
class VideoGAN:
    """视频GAN训练器"""
    def __init__(self, generator, discriminator, latent_dim, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device = device
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_videos):
        """训练一步"""
        batch_size = real_videos.size(0)
        
        # 标签
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # ==================== 训练判别器 ====================
        self.d_optimizer.zero_grad()
        
        # 真实视频
        real_output = self.discriminator(real_videos)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # 生成假视频
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_videos = self.generator(z).detach()
        fake_output = self.discriminator(fake_videos)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # ==================== 训练生成器 ====================
        self.g_optimizer.zero_grad()
        
        # 生成假视频
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_videos = self.generator(z)
        fake_output = self.discriminator(fake_videos)
        
        # 生成器损失
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_video(self, num_videos=1):
        """生成视频"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_videos, self.latent_dim).to(self.device)
            videos = self.generator(z)
        self.generator.train()
        return videos
 ```

## 实践项目

### 项目一：图像生成系统

**目标**：构建一个完整的图像生成系统，支持多种生成模式

**实现步骤**：

1. **数据准备**
```python
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CelebA

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到[-1, 1]
])

# 加载数据集
dataset = CelebA(root='./data', split='train', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

2. **模型训练**
```python
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型
generator = DCGANGenerator(latent_dim=100, num_channels=3)
discriminator = DCGANDiscriminator(num_channels=3)

# 应用权重初始化
generator.apply(weights_init)
discriminator.apply(weights_init)

# 创建GAN训练器
gan = GAN(generator, discriminator, latent_dim=100, device=device)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx, (real_data, _) in enumerate(dataloader):
        real_data = real_data.to(device)
        
        # 训练一步
        d_loss, g_loss = gan.train_step(real_data)
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}], '
                  f'D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}')
    
    # 每10个epoch生成样本
    if epoch % 10 == 0:
        samples = gan.generate_samples(16)
        # 保存样本图像
        save_image(samples, f'generated_epoch_{epoch}.png', nrow=4, normalize=True)
```

3. **评估指标**
```python
class GenerationMetrics:
    """生成质量评估指标"""
    
    @staticmethod
    def inception_score(images, batch_size=32, splits=10):
        """计算Inception Score"""
        from torchvision.models import inception_v3
        import torch.nn.functional as F
        
        # 加载预训练的Inception模型
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model.eval()
        
        # 预测
        preds = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                pred = inception_model(batch)
                pred = F.softmax(pred, dim=1)
                preds.append(pred)
        
        preds = torch.cat(preds, dim=0)
        
        # 计算IS
        scores = []
        for i in range(splits):
            part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
            kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0, keepdim=True)))
            kl = torch.mean(torch.sum(kl, dim=1))
            scores.append(torch.exp(kl))
        
        return torch.mean(torch.stack(scores)), torch.std(torch.stack(scores))
    
    @staticmethod
    def fid_score(real_images, fake_images):
        """计算Fréchet Inception Distance"""
        # 这里是简化版本，实际实现需要使用预训练的Inception网络特征
        from scipy.linalg import sqrtm
        import numpy as np
        
        # 提取特征（简化）
        def extract_features(images):
            # 实际应该使用Inception网络的特征层
            return images.view(images.size(0), -1).cpu().numpy()
        
        real_features = extract_features(real_images)
        fake_features = extract_features(fake_images)
        
        # 计算均值和协方差
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # 计算FID
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

# 评估生成质量
metrics = GenerationMetrics()

# 生成样本进行评估
with torch.no_grad():
    fake_samples = gan.generate_samples(1000)
    real_samples = next(iter(dataloader))[0][:1000]
    
    # 计算IS和FID
    is_mean, is_std = metrics.inception_score(fake_samples)
    fid = metrics.fid_score(real_samples, fake_samples)
    
    print(f'Inception Score: {is_mean:.2f} ± {is_std:.2f}')
    print(f'FID Score: {fid:.2f}')
```

### 项目二：文本到图像生成

**目标**：实现基于文本描述的图像生成系统

**核心代码**：
```python
class TextToImageGenerator:
    """文本到图像生成器"""
    def __init__(self, diffusion_model, text_encoder, device):
        self.diffusion_model = diffusion_model
        self.text_encoder = text_encoder
        self.device = device
    
    def encode_text(self, text_prompts):
        """编码文本提示"""
        # 使用CLIP或其他文本编码器
        tokens = clip.tokenize(text_prompts).to(self.device)
        text_features = self.text_encoder.encode_text(tokens)
        return text_features
    
    def generate_from_text(self, text_prompt, num_images=1, guidance_scale=7.5):
        """从文本生成图像"""
        # 编码文本
        text_features = self.encode_text([text_prompt] * num_images)
        
        # 生成图像
        images = []
        for i in range(num_images):
            # 使用扩散模型生成
            image = self.diffusion_model.sample_with_guidance(
                text_features[i:i+1], guidance_scale=guidance_scale
            )
            images.append(image)
        
        return torch.cat(images, dim=0)
    
    def interactive_generation(self):
        """交互式生成界面"""
        while True:
            prompt = input("请输入文本描述（输入'quit'退出）: ")
            if prompt.lower() == 'quit':
                break
            
            print("正在生成图像...")
            try:
                image = self.generate_from_text(prompt, num_images=1)
                
                # 保存图像
                timestamp = int(time.time())
                filename = f"generated_{timestamp}.png"
                save_image(image, filename, normalize=True)
                print(f"图像已保存为: {filename}")
                
            except Exception as e:
                print(f"生成失败: {e}")

# 使用示例
text_to_image = TextToImageGenerator(diffusion_model, clip_model, device)
text_to_image.interactive_generation()
```

### 项目三：音乐生成系统

**目标**：构建能够生成音乐的AI系统

**实现框架**：
```python
class MusicGenerator:
    """音乐生成系统"""
    def __init__(self, model, sample_rate=22050):
        self.model = model
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate)
    
    def generate_melody(self, length_seconds=10, temperature=1.0):
        """生成旋律"""
        length_samples = int(length_seconds * self.sample_rate)
        
        # 使用WaveNet生成
        generated_audio = self.model.generate(length_samples, temperature)
        
        # 后处理
        audio = self.audio_processor.mu_law_decode(generated_audio)
        
        return audio
    
    def generate_with_style(self, style_prompt, length_seconds=10):
        """根据风格提示生成音乐"""
        # 这里可以集成条件生成
        # 例如使用文本编码器编码风格描述
        style_embedding = self.encode_style(style_prompt)
        
        # 条件生成
        generated_audio = self.model.conditional_generate(
            style_embedding, length_seconds * self.sample_rate
        )
        
        return generated_audio
    
    def save_audio(self, audio_tensor, filename):
        """保存音频文件"""
        import soundfile as sf
        
        # 转换为numpy数组
        audio_np = audio_tensor.squeeze().cpu().numpy()
        
        # 保存为WAV文件
        sf.write(filename, audio_np, self.sample_rate)
        print(f"音频已保存为: {filename}")

# 使用示例
music_gen = MusicGenerator(wavenet_model)

# 生成不同风格的音乐
styles = ["classical", "jazz", "electronic", "folk"]
for style in styles:
    audio = music_gen.generate_with_style(f"{style} music", length_seconds=15)
    music_gen.save_audio(audio, f"{style}_generated.wav")
```

### 项目四：创意写作助手

**目标**：开发AI创意写作助手，支持多种文本生成任务

**核心功能**：
```python
class CreativeWritingAssistant:
    """创意写作助手"""
    def __init__(self, language_model, tokenizer):
        self.model = language_model
        self.tokenizer = tokenizer
    
    def generate_story(self, prompt, max_length=500, temperature=0.8):
        """生成故事"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story[len(prompt):].strip()
    
    def generate_poetry(self, theme, style="free verse"):
        """生成诗歌"""
        prompt = f"Write a {style} poem about {theme}:\n"
        
        poem = self.generate_story(prompt, max_length=200, temperature=0.9)
        return poem
    
    def continue_writing(self, existing_text, continuation_length=200):
        """续写文本"""
        continuation = self.generate_story(
            existing_text, 
            max_length=len(existing_text.split()) + continuation_length,
            temperature=0.7
        )
        return continuation
    
    def writing_workshop(self):
        """写作工作坊界面"""
        print("欢迎来到AI创意写作工作坊！")
        
        while True:
            print("\n选择功能:")
            print("1. 故事生成")
            print("2. 诗歌创作")
            print("3. 文本续写")
            print("4. 退出")
            
            choice = input("请选择 (1-4): ")
            
            if choice == '1':
                prompt = input("请输入故事开头: ")
                story = self.generate_story(prompt)
                print(f"\n生成的故事:\n{story}")
                
            elif choice == '2':
                theme = input("请输入诗歌主题: ")
                style = input("请输入诗歌风格 (默认自由诗): ") or "free verse"
                poem = self.generate_poetry(theme, style)
                print(f"\n生成的诗歌:\n{poem}")
                
            elif choice == '3':
                text = input("请输入需要续写的文本: ")
                continuation = self.continue_writing(text)
                print(f"\n续写结果:\n{continuation}")
                
            elif choice == '4':
                print("感谢使用AI创意写作工作坊！")
                break
            
            else:
                print("无效选择，请重试。")

# 使用示例
writing_assistant = CreativeWritingAssistant(gpt_model, tokenizer)
writing_assistant.writing_workshop()
```

## 学习评估

### 理论评估

1. **概念理解**（25分）
   - 生成模型的数学基础
   - VAE的重参数化技巧
   - GAN的对抗训练原理
   - 扩散模型的前向和逆向过程

2. **算法分析**（25分）
   - 不同生成模型的优缺点比较
   - 训练稳定性问题及解决方案
   - 生成质量评估指标
   - 条件生成的实现方法

### 实践评估

1. **编程实现**（30分）
   - 实现基础的VAE模型
   - 构建GAN训练循环
   - 扩散模型的采样过程
   - 多模态生成系统

2. **项目作品**（20分）
   - 创意生成项目的完整性
   - 生成质量和多样性
   - 用户交互设计
   - 技术创新点

### 综合评估标准

- **优秀（90-100分）**：深入理解生成式AI原理，能够独立实现复杂的生成系统，作品具有高度创新性
- **良好（80-89分）**：较好掌握主要概念和技术，能够实现标准的生成模型，作品质量良好
- **中等（70-79分）**：基本理解核心概念，能够在指导下完成实践项目
- **及格（60-69分）**：了解基本概念，能够运行和修改现有代码

## 延伸学习

### 前沿研究方向

1. **大规模生成模型**
   - GPT系列语言模型
   - DALL-E图像生成
   - Stable Diffusion
   - Midjourney艺术生成

2. **可控生成**
   - 风格迁移
   - 属性编辑
   - 语义引导生成
   - 交互式生成

3. **多模态生成**
   - 文本到图像
   - 图像到文本
   - 音频到视频
   - 跨模态转换

4. **生成模型的应用**
   - 内容创作
   - 数据增强
   - 艺术创作
   - 游戏开发

### 推荐资源

**学术论文**：
- "Generative Adversarial Networks" (Goodfellow et al., 2014)
- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Attention Is All You Need" (Vaswani et al., 2017)

**开源项目**：
- Hugging Face Diffusers
- OpenAI CLIP
- StyleGAN系列
- WaveNet实现

**在线课程**：
- CS236: Deep Generative Models (Stanford)
- Deep Learning Specialization (Coursera)
- Generative AI with Large Language Models

## 模块总结

本模块深入探讨了生成式AI与创意计算的核心技术和应用。我们学习了：

1. **理论基础**：掌握了生成模型的数学原理，包括概率论、信息论和变分推理

2. **核心技术**：深入理解了VAE、GAN、扩散模型等主流生成技术的原理和实现

3. **多模态生成**：学习了文本、图像、音频、视频等不同模态的生成方法

4. **实践应用**：通过四个综合项目，掌握了生成式AI在创意领域的实际应用

5. **质量评估**：了解了生成质量的评估指标和方法

生成式AI正在革命性地改变内容创作、艺术设计、娱乐产业等多个领域。随着技术的不断发展，我们期待看到更多创新的应用和突破。掌握这些技术不仅有助于理解AI的创造能力，也为未来的研究和应用奠定了坚实基础。

**下一步学习建议**：
- 深入研究最新的大规模生成模型
- 探索生成式AI的伦理和社会影响
- 参与开源项目，贡献自己的创新想法
- 关注产业应用，了解商业化趋势