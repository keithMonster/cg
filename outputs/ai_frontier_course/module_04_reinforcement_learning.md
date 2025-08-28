# 模块四：强化学习与智能决策

## 课程信息
- **模块编号**：Module 04
- **模块名称**：强化学习与智能决策
- **学时安排**：理论课程 12学时，实践课程 10学时
- **学习目标**：掌握强化学习的核心理论、算法实现和实际应用

## 第一章：强化学习基础理论

### 1.1 什么是强化学习？

🎮 **简单理解**：强化学习就像教一个小孩玩游戏，通过奖励和惩罚让他学会做出最好的决策。

**生活中的强化学习例子**：
- 🎮 **玩游戏**：通过不断试错，学会如何获得高分
- 🚗 **学开车**：通过练习，学会在不同路况下做出正确判断
- 🐕 **训练宠物**：用零食奖励，教狗狗学会各种技能
- 📈 **股票投资**：通过盈亏经验，学会更好的投资策略
- 🤖 **机器人走路**：通过摔倒和成功，学会保持平衡

### 1.2 强化学习的核心要素（用游戏来理解）

🎯 **想象你在玩一个闯关游戏**：

#### 🗺️ 状态（State）- 游戏当前的情况

**简单理解**：就是游戏中"现在是什么情况"。

**生活例子**：
- 🎮 **游戏中**：角色的位置、血量、装备、敌人位置
- 🚗 **开车时**：车速、路况、红绿灯、周围车辆
- 🏠 **扫地机器人**：当前位置、电量、哪里已经扫过
- 📱 **手机推荐**：用户喜好、时间、地点、历史行为

**关键特点**：
- 📊 **包含决策需要的所有信息**
- 🔄 **会随着行动而改变**
- 📍 **就像游戏的"存档点"

#### 🎯 动作（Action）- 可以做什么

**简单理解**：在当前情况下，你可以选择做的事情。

**生活例子**：
- 🎮 **游戏中**：向左走、向右走、跳跃、攻击
- 🚗 **开车时**：加速、刹车、转弯、变道
- 🏠 **扫地机器人**：前进、后退、左转、右转、开始清扫
- 📱 **推荐系统**：推荐电影A、推荐电影B、不推荐

**关键特点**：
- 🎲 **每个状态下可能有不同的可选动作**
- ⚡ **动作会改变当前状态**
- 🎯 **选择不同动作会有不同结果**

#### 🎁 奖励（Reward）- 做得好不好的反馈

**简单理解**：就像游戏中的得分，告诉你这个动作是好是坏。

**生活例子**：
- 🎮 **游戏中**：击败敌人+100分，被击中-50分
- 🚗 **开车时**：安全到达+奖励，违章-惩罚
- 🏠 **扫地机器人**：清扫干净+奖励，撞墙-惩罚
- 📱 **推荐系统**：用户点赞+奖励，用户跳过-惩罚

**奖励的特点**：
- ✅ **正奖励**：鼓励这样做
- ❌ **负奖励（惩罚）**：不要这样做
- 🎯 **指导学习方向**：追求高奖励，避免惩罚

#### 🔄 环境转移（Environment Transition）- 世界如何变化

**简单理解**：当你做了一个动作后，世界会如何变化。

**生活例子**：
- 🎮 **游戏中**：按"向右"键 → 角色向右移动一格
- 🚗 **开车时**：踩油门 → 车速增加，位置前进
- 🏠 **扫地机器人**：选择"前进" → 移动到前方位置
- 🎲 **有时有随机性**：同样的动作可能有不同结果

#### ⏰ 折扣因子（Discount Factor）- 未来有多重要

**简单理解**：现在的奖励和未来的奖励，哪个更重要？

**生活比喻**：
- 💰 **金钱的时间价值**：今天的100元比明年的100元更值钱
- 🍰 **即时满足 vs 长远利益**：现在吃蛋糕 vs 保持健康
- 🎯 **游戏策略**：追求即时得分 vs 为最终胜利布局

**不同的折扣因子**：
- 🔥 **折扣因子 = 0**：只关心现在，"今朝有酒今朝醉"
- ⚖️ **折扣因子 = 0.9**：未来也重要，但现在更重要
- 🔮 **折扣因子 = 1**：现在和未来同等重要

#### 马尔可夫性质

**马尔可夫性质定义**
```
Pr{Sₜ₊₁ = s' | Sₜ = s, Sₜ₋₁ = sₜ₋₁, ..., S₀ = s₀} = Pr{Sₜ₊₁ = s' | Sₜ = s}
```
- 未来状态只依赖当前状态
- 与历史路径无关
- 简化了问题的复杂性

**状态的充分性**
- 状态包含所有相关的历史信息
- 能够做出最优决策的最小信息集
- 在实际应用中可能需要状态增强

**非马尔可夫环境的处理**
- 状态增强：包含历史信息
- 循环神经网络：记忆机制
- 部分可观测MDP：信念状态

#### 策略与价值函数

**策略（Policy）**

*确定性策略*
```
π(s) = a
```
- 每个状态对应唯一动作
- 简单但可能不是最优

*随机策略*
```
π(a|s) = Pr{Aₜ = a | Sₜ = s}
```
- 每个状态对应动作的概率分布
- 更灵活，可以处理不确定性
- 满足：∑ₐ π(a|s) = 1

**状态价值函数**
```
Vᵖ(s) = E_π[Gₜ | Sₜ = s]
     = E_π[∑_{k=0}^∞ γᵏRₜ₊ₖ₊₁ | Sₜ = s]
```
- 从状态s开始，遵循策略π的期望累积奖励
- 评估状态的好坏
- 递归关系：Bellman方程

**动作价值函数**
```
Qᵖ(s,a) = E_π[Gₜ | Sₜ = s, Aₜ = a]
        = E_π[∑_{k=0}^∞ γᵏRₜ₊ₖ₊₁ | Sₜ = s, Aₜ = a]
```
- 在状态s执行动作a，然后遵循策略π的期望累积奖励
- 评估状态-动作对的价值
- 与状态价值函数的关系：Vᵖ(s) = ∑ₐ π(a|s)Qᵖ(s,a)

### 1.2 Bellman方程

#### Bellman期望方程

**状态价值函数的Bellman方程**
```
Vᵖ(s) = ∑ₐ π(a|s) ∑ₛ' P(s'|s,a)[R(s,a,s') + γVᵖ(s')]
```
- 当前状态价值等于即时奖励加上后续状态价值的期望
- 递归定义，体现了价值的传播
- 线性方程组，有唯一解

**动作价值函数的Bellman方程**
```
Qᵖ(s,a) = ∑ₛ' P(s'|s,a)[R(s,a,s') + γ ∑ₐ' π(a'|s')Qᵖ(s',a')]
```
- 动作价值的递归定义
- 连接当前动作和未来策略

#### Bellman最优方程

**最优状态价值函数**
```
V*(s) = max_π Vᵖ(s)
      = max_a ∑ₛ' P(s'|s,a)[R(s,a,s') + γV*(s')]
```
- 所有策略中的最大状态价值
- 最优策略的存在性保证

**最优动作价值函数**
```
Q*(s,a) = ∑ₛ' P(s'|s,a)[R(s,a,s') + γ max_a' Q*(s',a')]
```
- 最优策略下的动作价值
- 与最优状态价值的关系：V*(s) = max_a Q*(s,a)

**最优策略**
```
π*(s) = argmax_a Q*(s,a)
```
- 贪婪地选择最优动作价值对应的动作
- 可能存在多个最优策略
- 所有最优策略具有相同的价值函数

#### 策略改进定理

**策略评估**
- 给定策略π，计算Vᵖ(s)
- 解Bellman期望方程
- 迭代方法或直接求解

**策略改进**
```
π'(s) = argmax_a ∑ₛ' P(s'|s,a)[R(s,a,s') + γVᵖ(s')]
```
- 基于当前价值函数的贪婪策略
- 保证策略不会变差：Vᵖ'(s) ≥ Vᵖ(s)

**策略迭代算法**
1. 初始化策略π₀
2. 策略评估：计算Vᵖᵢ
3. 策略改进：πᵢ₊₁ = greedy(Vᵖᵢ)
4. 重复直到收敛

### 1.3 动态规划方法

#### 价值迭代

**算法原理**
```
Vₖ₊₁(s) = max_a ∑ₛ' P(s'|s,a)[R(s,a,s') + γVₖ(s')]
```
- 直接迭代Bellman最优方程
- 不需要显式的策略
- 收敛到最优价值函数

**算法步骤**
1. 初始化V₀(s) = 0 for all s
2. 对所有状态更新价值：Vₖ₊₁(s) = max_a ...
3. 检查收敛：||Vₖ₊₁ - Vₖ|| < ε
4. 提取最优策略：π*(s) = argmax_a ...

**收敛性分析**
- 压缩映射定理保证收敛
- 收敛速度：O(γᵏ)
- 计算复杂度：O(|S|²|A|)每次迭代

#### 策略迭代

**算法流程**

*策略评估阶段*
```
Vᵖᵢ(s) = ∑ₐ πᵢ(a|s) ∑ₛ' P(s'|s,a)[R(s,a,s') + γVᵖᵢ(s')]
```
- 解线性方程组
- 或使用迭代方法近似

*策略改进阶段*
```
πᵢ₊₁(s) = argmax_a ∑ₛ' P(s'|s,a)[R(s,a,s') + γVᵖᵢ(s')]
```
- 贪婪策略更新
- 保证单调改进

**算法特点**
- 每次迭代策略都有改进
- 有限步内收敛到最优策略
- 计算量大但收敛快

#### 截断策略迭代

**动机**
- 策略评估不需要完全收敛
- 平衡计算效率和精度
- 介于价值迭代和策略迭代之间

**算法设计**
- 策略评估进行k步迭代
- k=1时退化为价值迭代
- k=∞时等价于策略迭代

**优势**
- 灵活调节计算量
- 通常k=3-5就足够
- 实际应用中的标准方法

## 第二章：无模型强化学习

### 2.1 蒙特卡洛方法

#### 基本思想

**采样估计**
- 通过经验来估计价值函数
- 不需要环境模型
- 基于完整的episode

**大数定律**
```
Vᵖ(s) = E_π[Gₜ | Sₜ = s] ≈ (1/n) ∑ᵢ₌₁ⁿ Gₜ⁽ⁱ⁾
```
- 样本均值收敛到期望值
- 需要大量样本
- 无偏估计

#### 首次访问MC

**算法步骤**
1. 初始化：V(s) = 0, Returns(s) = []
2. 对每个episode：
   - 生成episode：S₀, A₀, R₁, S₁, A₁, R₂, ...
   - 对每个状态s（首次出现）：
     - G = 从s开始的累积奖励
     - Returns(s).append(G)
     - V(s) = average(Returns(s))

**特点**
- 每个episode中每个状态只考虑一次
- 避免同一episode内的相关性
- 收敛到真实价值函数

#### 每次访问MC

**算法修改**
- 每次访问状态s都进行更新
- 不区分是否首次访问
- 同样收敛到真实价值函数

**在线更新**
```
V(s) ← V(s) + α[G - V(s)]
```
- α：学习率
- 增量式更新
- 适合在线学习

#### MC控制

**策略评估**
- 使用MC方法估计Qᵖ(s,a)
- 需要探索所有状态-动作对
- 探索与利用的平衡

**ε-贪婪策略**
```
π(a|s) = {
  1-ε+ε/|A(s)|  if a = argmax_a Q(s,a)
  ε/|A(s)|      otherwise
}
```
- 以概率ε随机选择动作
- 以概率1-ε选择最优动作
- 保证持续探索

**GLIE（Greedy in the Limit with Infinite Exploration）**
- 所有状态-动作对被无限次访问
- 策略在极限情况下变为贪婪
- 保证收敛到最优策略

### 2.2 时序差分学习

#### TD(0)算法

**基本思想**
- 结合MC和DP的优点
- 使用bootstrap：基于估计更新估计
- 不需要等待episode结束

**TD更新规则**
```
V(Sₜ) ← V(Sₜ) + α[Rₜ₊₁ + γV(Sₜ₊₁) - V(Sₜ)]
```
- TD目标：Rₜ₊₁ + γV(Sₜ₊₁)
- TD误差：δₜ = Rₜ₊₁ + γV(Sₜ₊₁) - V(Sₜ)
- 在线学习，每步更新

**与MC的比较**

*MC方法*
- 无偏估计
- 高方差
- 需要完整episode
- 不使用bootstrap

*TD方法*
- 有偏估计（初期）
- 低方差
- 在线更新
- 使用bootstrap

#### SARSA算法

**算法名称**
- State-Action-Reward-State-Action
- 在策略（On-policy）方法
- 学习当前策略的价值

**更新规则**
```
Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + α[Rₜ₊₁ + γQ(Sₜ₊₁,Aₜ₊₁) - Q(Sₜ,Aₜ)]
```
- 使用实际执行的动作Aₜ₊₁
- 学习当前策略的Q函数
- 保守的学习方式

**算法流程**
1. 初始化Q(s,a)
2. 对每个episode：
   - 初始化S
   - 选择A（基于Q的ε-贪婪）
   - 重复：
     - 执行A，观察R, S'
     - 选择A'（基于Q的ε-贪婪）
     - 更新Q(S,A)
     - S ← S', A ← A'

#### Q-Learning算法

**离策略学习**
- 学习最优策略的价值
- 行为策略可以不同于目标策略
- 更激进的学习方式

**更新规则**
```
Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + α[Rₜ₊₁ + γ max_a Q(Sₜ₊₁,a) - Q(Sₜ,Aₜ)]
```
- 使用max操作而非实际动作
- 直接学习最优Q函数
- 收敛到Q*

**算法特点**
- 简单且强大
- 不依赖于行为策略
- 可以从任意策略的数据中学习
- 广泛应用的基础算法

#### Expected SARSA

**期望更新**
```
Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + α[Rₜ₊₁ + γ ∑_a π(a|Sₜ₊₁)Q(Sₜ₊₁,a) - Q(Sₜ,Aₜ)]
```
- 使用期望而非采样
- 减少方差
- 更稳定的学习

**优势**
- 比SARSA方差更小
- 比Q-learning更稳定
- 可以处理随机策略

### 2.3 多步时序差分

#### n步TD方法

**n步回报**
```
Gₜ⁽ⁿ⁾ = Rₜ₊₁ + γRₜ₊₂ + ... + γⁿ⁻¹Rₜ₊ₙ + γⁿV(Sₜ₊ₙ)
```
- 结合n步实际奖励和估计价值
- n=1：TD(0)
- n=∞：MC方法

**n步TD更新**
```
V(Sₜ) ← V(Sₜ) + α[Gₜ⁽ⁿ⁾ - V(Sₜ)]
```
- 平衡偏差和方差
- 需要延迟n步更新

#### TD(λ)算法

**资格迹（Eligibility Traces）**
```
eₜ(s) = {
  γλeₜ₋₁(s) + 1  if s = Sₜ
  γλeₜ₋₁(s)      otherwise
}
```
- 记录状态的访问历史
- λ：衰减参数
- 结合频率和新近性

**TD(λ)更新**
```
δₜ = Rₜ₊₁ + γV(Sₜ₊₁) - V(Sₜ)
V(s) ← V(s) + αδₜeₜ(s), ∀s
```
- 所有状态同时更新
- 更新量与资格迹成正比
- 统一了TD和MC方法

**λ参数的作用**
- λ=0：TD(0)
- λ=1：MC方法
- 0<λ<1：平衡偏差和方差

#### SARSA(λ)

**动作价值的资格迹**
```
eₜ(s,a) = {
  γλeₜ₋₁(s,a) + 1  if s = Sₜ, a = Aₜ
  γλeₜ₋₁(s,a)      otherwise
}
```

**更新规则**
```
δₜ = Rₜ₊₁ + γQ(Sₜ₊₁,Aₜ₊₁) - Q(Sₜ,Aₜ)
Q(s,a) ← Q(s,a) + αδₜeₜ(s,a), ∀s,a
```

**替换迹（Replacing Traces）**
```
eₜ(s,a) = {
  1                if s = Sₜ, a = Aₜ
  γλeₜ₋₁(s,a)      otherwise
}
```
- 避免同一状态-动作对的累积
- 通常性能更好

## 第三章：深度强化学习

### 3.1 价值函数近似

#### 函数近似的必要性

**维度诅咒**
- 状态空间过大或连续
- 无法存储所有状态的价值
- 需要泛化能力

**函数近似器**
- 线性函数：V(s) = θᵀφ(s)
- 神经网络：V(s) = fθ(s)
- 决策树、核方法等

**特征工程**
- 状态表示的重要性
- 手工设计 vs 自动学习
- 深度学习的优势

#### 深度Q网络（DQN）

**基本架构**
```
Q(s,a;θ) = Neural_Network(s,a;θ)
```
- 输入：状态s（和动作a）
- 输出：Q值
- 参数：神经网络权重θ

**损失函数**
```
L(θ) = E[(yᵢ - Q(sᵢ,aᵢ;θ))²]
其中 yᵢ = rᵢ + γ max_a' Q(s'ᵢ,a';θ⁻)
```
- 均方误差损失
- 目标网络θ⁻
- 梯度下降优化

**关键技术**

*经验回放（Experience Replay）*
- 存储经验：(s,a,r,s')
- 随机采样训练
- 打破数据相关性
- 提高样本效率

*目标网络（Target Network）*
- 固定目标Q网络参数
- 周期性更新：θ⁻ ← θ
- 稳定训练过程
- 减少目标的变化

#### DQN算法实现

**算法流程**
1. 初始化经验池D和Q网络
2. 对每个episode：
   - 观察状态s
   - 选择动作a（ε-贪婪）
   - 执行a，观察r,s'
   - 存储(s,a,r,s')到D
   - 从D采样batch训练Q网络
   - 周期性更新目标网络

**网络结构**
```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

#### DQN的改进

**Double DQN**

*问题*
- Q-learning的过估计偏差
- max操作导致的正偏差
- 影响学习稳定性

*解决方案*
```
yᵢ = rᵢ + γQ(s'ᵢ, argmax_a Q(s'ᵢ,a;θ); θ⁻)
```
- 用在线网络选择动作
- 用目标网络评估价值
- 减少过估计

**Dueling DQN**

*网络架构*
```
Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```
- 分离状态价值和优势函数
- V(s)：状态价值流
- A(s,a)：优势函数流
- 更好的价值估计

**优先经验回放**

*TD误差优先级*
```
P(i) = |δᵢ| + ε
```
- 根据TD误差采样
- 重要样本更频繁训练
- 提高学习效率

*重要性采样权重*
```
wᵢ = (1/N · 1/P(i))^β
```
- 修正采样偏差
- β从0逐渐增加到1

### 3.2 策略梯度方法

#### 策略参数化

**随机策略**
```
π(a|s;θ) = 策略网络输出的概率分布
```
- 直接参数化策略
- 可以处理连续动作空间
- 自然的探索机制

**策略梯度定理**
```
∇_θ J(θ) = E_π[∇_θ log π(a|s;θ) Q^π(s,a)]
```
- J(θ)：策略性能
- 梯度方向指向更好的策略
- 无需知道环境模型

#### REINFORCE算法

**基本思想**
- 蒙特卡洛策略梯度
- 使用完整episode的回报
- 无偏但高方差

**算法步骤**
1. 用当前策略生成episode
2. 计算每步的回报Gₜ
3. 更新策略参数：
   ```
   θ ← θ + α∇_θ log π(Aₜ|Sₜ;θ) Gₜ
   ```

**基线减方差**
```
∇_θ J(θ) = E_π[∇_θ log π(a|s;θ) (Q^π(s,a) - b(s))]
```
- b(s)：基线函数
- 通常使用状态价值V^π(s)
- 减少方差但保持无偏

#### Actor-Critic方法

**基本架构**
- Actor：策略网络π(a|s;θ)
- Critic：价值网络V(s;w)
- 结合策略梯度和价值函数近似

**优势函数**
```
A(s,a) = Q(s,a) - V(s)
      ≈ r + γV(s') - V(s)
```
- 衡量动作相对于平均水平的优势
- 减少方差
- TD误差作为优势估计

**算法更新**

*Critic更新*
```
w ← w + α_w δ ∇_w V(s;w)
其中 δ = r + γV(s';w) - V(s;w)
```

*Actor更新*
```
θ ← θ + α_θ δ ∇_θ log π(a|s;θ)
```

#### A3C算法

**异步优势Actor-Critic**
- 多个并行worker
- 异步更新全局参数
- 提高样本效率和稳定性

**算法特点**
- 不需要经验回放
- 在线学习
- 并行加速
- 更好的探索

**实现要点**
```python
# Worker线程
def worker(global_model, optimizer, worker_id):
    local_model = copy.deepcopy(global_model)
    
    while True:
        # 收集经验
        states, actions, rewards = collect_experience()
        
        # 计算损失
        actor_loss, critic_loss = compute_loss()
        
        # 更新全局模型
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### 3.3 高级深度强化学习

#### PPO算法

**信任域方法**
- 限制策略更新幅度
- 避免破坏性更新
- 稳定训练过程

**重要性采样**
```
L^CPI(θ) = E_t[π_θ(a_t|s_t)/π_θ_old(a_t|s_t) A_t]
```
- 使用旧策略的数据
- 重要性权重修正
- 提高样本效率

**裁剪目标函数**
```
L^CLIP(θ) = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
其中 r_t(θ) = π_θ(a_t|s_t)/π_θ_old(a_t|s_t)
```
- 裁剪重要性权重
- 防止过大的策略更新
- 简单且有效

**算法优势**
- 实现简单
- 性能稳定
- 广泛应用
- 超参数鲁棒

#### SAC算法

**软Actor-Critic**
- 最大熵强化学习
- 平衡性能和探索
- 连续控制的利器

**目标函数**
```
J(π) = E_π[∑_t r(s_t,a_t) + α H(π(·|s_t))]
```
- H(π)：策略熵
- α：温度参数
- 鼓励探索

**软Bellman方程**
```
V(s) = E_a~π[Q(s,a) - α log π(a|s)]
Q(s,a) = r + γE_s'[V(s')]
```
- 包含熵项的价值函数
- 自动平衡探索和利用

**算法特点**
- 样本效率高
- 稳定训练
- 自适应探索
- 适合连续控制

#### DDPG算法

**确定性策略梯度**
```
∇_θ J ≈ E_s[∇_θ π(s|θ) ∇_a Q(s,a|φ)|_{a=π(s|θ)}]
```
- 确定性策略：μ(s|θ)
- 连续动作空间
- 结合DQN和策略梯度

**关键技术**

*目标网络*
- Actor和Critic都有目标网络
- 软更新：θ' ← τθ + (1-τ)θ'
- 稳定训练

*噪声探索*
```
a = μ(s|θ) + N(0,σ²)
```
- 添加噪声进行探索
- Ornstein-Uhlenbeck过程
- 时间相关的噪声

#### TD3算法

**Twin Delayed DDPG**
- DDPG的改进版本
- 解决过估计问题
- 提高稳定性

**关键改进**

*双Critic网络*
```
y = r + γ min(Q_1(s',a'), Q_2(s',a'))
```
- 使用较小的Q值作为目标
- 减少过估计偏差

*延迟策略更新*
- Critic更新频率高于Actor
- 减少策略更新的方差
- 更稳定的学习

*目标策略平滑*
```
a' = μ(s') + clip(ε, -c, c)
```
- 目标动作添加噪声
- 平滑价值估计
- 提高鲁棒性

## 第四章：多智能体强化学习

### 4.1 多智能体环境

#### 基本概念

**多智能体系统**
- 多个智能体同时学习和决策
- 环境的非平稳性
- 智能体间的相互影响

**博弈论基础**

*纳什均衡*
- 每个智能体的策略都是对其他智能体策略的最佳响应
- 稳定的策略组合
- 可能存在多个均衡

*帕累托最优*
- 无法在不损害某个智能体的情况下改善其他智能体
- 社会最优解
- 可能与纳什均衡不同

#### 合作与竞争

**合作设置**
- 共同目标
- 信息共享
- 协调行动
- 团队奖励

**竞争设置**
- 对抗目标
- 零和博弈
- 策略隐藏
- 个体奖励

**混合设置**
- 部分合作，部分竞争
- 联盟形成
- 动态关系
- 复杂策略

### 4.2 独立学习

**基本思想**
- 每个智能体独立学习
- 将其他智能体视为环境的一部分
- 简单但可能不稳定

**算法应用**
- 独立Q-learning
- 独立策略梯度
- 独立Actor-Critic

**挑战**
- 环境非平稳性
- 收敛性无保证
- 可能陷入次优解

### 4.3 中心化训练分布式执行

**CTDE框架**
- 训练时可以访问全局信息
- 执行时只使用局部信息
- 平衡性能和实用性

**MADDPG算法**

*中心化Critic*
```
Q_i(s_1,...,s_N, a_1,...,a_N)
```
- 使用所有智能体的状态和动作
- 更准确的价值估计
- 稳定训练

*分布式Actor*
```
π_i(a_i|s_i)
```
- 只使用局部观察
- 独立执行
- 满足部分可观察约束

**QMIX算法**

*价值分解*
```
Q_tot(s,a) = f(Q_1(s_1,a_1), ..., Q_N(s_N,a_N))
```
- 单调性约束：∂Q_tot/∂Q_i ≥ 0
- 保证个体和团队目标一致
- 混合网络学习分解函数

## 实践项目

### 项目一：经典控制问题

**CartPole平衡**

*环境描述*
- 状态：位置、速度、角度、角速度
- 动作：向左或向右推车
- 目标：保持杆子平衡

*算法实现*
```python
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.memory = ReplayBuffer(10000)
        self.epsilon = 1.0
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def learn(self):
        if len(self.memory) < 1000:
            return
        
        batch = self.memory.sample(32)
        states, actions, rewards, next_states, dones = batch
        
        current_q = self.q_network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (0.99 * next_q * (1 - dones))
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 项目二：Atari游戏

**环境预处理**
```python
class AtariWrapper:
    def __init__(self, env):
        self.env = env
        self.frame_stack = deque(maxlen=4)
    
    def preprocess(self, frame):
        # 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 缩放
        resized = cv2.resize(gray, (84, 84))
        return resized / 255.0
    
    def step(self, action):
        total_reward = 0
        for _ in range(4):  # 跳帧
            frame, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        processed_frame = self.preprocess(frame)
        self.frame_stack.append(processed_frame)
        state = np.stack(self.frame_stack, axis=0)
        
        return state, total_reward, done, info
```

**CNN网络结构**
```python
class AtariDQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### 项目三：连续控制

**MuJoCo环境**
- 物理仿真环境
- 连续状态和动作空间
- 复杂的机器人控制任务

**DDPG实现**
```python
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        self.memory = ReplayBuffer(1000000)
        self.noise = OUNoise(action_dim)
    
    def act(self, state, add_noise=True):
        action = self.actor(state)
        if add_noise:
            action += self.noise.sample()
        return action.clamp(-self.max_action, self.max_action)
    
    def learn(self):
        batch = self.memory.sample(256)
        state, action, reward, next_state, done = batch
        
        # 更新Critic
        target_action = self.target_actor(next_state)
        target_q = self.target_critic(next_state, target_action)
        target_q = reward + (0.99 * target_q * (1 - done))
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.target_actor, self.actor, 0.001)
        self.soft_update(self.target_critic, self.critic, 0.001)
```

### 项目四：多智能体协作

**环境设计**
```python
class MultiAgentEnv:
    def __init__(self, n_agents=3):
        self.n_agents = n_agents
        self.agents = [Agent(i) for i in range(n_agents)]
        self.state_dim = 10
        self.action_dim = 5
    
    def step(self, actions):
        # 所有智能体同时执行动作
        rewards = []
        next_states = []
        
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            reward, next_state = agent.step(action, self.get_global_state())
            rewards.append(reward)
            next_states.append(next_state)
        
        return next_states, rewards, self.is_done(), {}
    
    def get_global_state(self):
        return np.concatenate([agent.get_state() for agent in self.agents])
```

**MADDPG实现**
```python
class MADDPGAgent:
    def __init__(self, agent_id, state_dim, action_dim, n_agents):
        self.agent_id = agent_id
        self.n_agents = n_agents
        
        # 局部Actor
        self.actor = Actor(state_dim, action_dim)
        # 全局Critic
        self.critic = Critic(state_dim * n_agents, action_dim * n_agents)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
    
    def act(self, state):
        return self.actor(state)
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # 全局状态和动作
        global_states = torch.cat(states, dim=1)
        global_actions = torch.cat(actions, dim=1)
        global_next_states = torch.cat(next_states, dim=1)
        
        # 目标动作
        target_actions = []
        for i, next_state in enumerate(next_states):
            target_action = self.target_actors[i](next_state)
            target_actions.append(target_action)
        global_target_actions = torch.cat(target_actions, dim=1)
        
        # 更新Critic
        target_q = self.target_critic(global_next_states, global_target_actions)
        target_q = rewards[self.agent_id] + 0.99 * target_q * (1 - dones[self.agent_id])
        
        current_q = self.critic(global_states, global_actions)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        # 更新Actor
        predicted_actions = actions.copy()
        predicted_actions[self.agent_id] = self.actor(states[self.agent_id])
        global_predicted_actions = torch.cat(predicted_actions, dim=1)
        
        actor_loss = -self.critic(global_states, global_predicted_actions).mean()
```

## 学习评估

### 理论理解评估

**1. MDP基础**
- 马尔可夫决策过程的数学定义
- Bellman方程的推导和理解
- 策略、价值函数的概念
- 最优性原理

**2. 算法原理**
- 动态规划方法的收敛性
- 蒙特卡洛和时序差分的区别
- 策略梯度定理的证明
- 函数近似的挑战

**3. 深度强化学习**
- DQN的关键技术及其作用
- Actor-Critic方法的优势
- 各种算法的适用场景
- 多智能体学习的复杂性

### 实践能力评估

**1. 算法实现**
- 从零实现经典RL算法
- 使用深度学习框架
- 调试和优化代码
- 处理实际环境

**2. 实验设计**
- 选择合适的算法和超参数
- 设计对比实验
- 分析学习曲线
- 评估算法性能

**3. 问题解决**
- 诊断训练问题
- 改进算法性能
- 适应新环境
- 扩展到复杂任务

### 应用能力评估

**1. 环境建模**
- 将实际问题转化为MDP
- 设计状态和动作空间
- 定义奖励函数
- 处理部分可观察性

**2. 算法选择**
- 根据问题特点选择算法
- 考虑计算资源限制
- 平衡性能和效率
- 处理连续和离散空间

**3. 系统集成**
- 将RL集成到实际系统
- 处理实时性要求
- 安全性考虑
- 可解释性需求

## 延伸学习

### 前沿研究方向

**模型基础强化学习**
- 世界模型学习
- 模型预测控制
- Dyna-Q算法
- 想象增强智能体

**分层强化学习**
- 选项框架
- 目标条件强化学习
- 元学习
- 课程学习

**安全强化学习**
- 约束优化
- 风险敏感学习
- 鲁棒性保证
- 可验证的RL

**离线强化学习**
- 批量强化学习
- 保守Q学习
- 行为克隆
- 分布偏移问题

### 应用领域

**游戏AI**
- AlphaGo/AlphaZero
- 实时策略游戏
- 多人在线游戏
- 程序化内容生成

**机器人控制**
- 运动控制
- 操作规划
- 导航
- 人机协作

**自动驾驶**
- 路径规划
- 决策制定
- 多车协调
- 安全保证

**金融交易**
- 算法交易
- 投资组合优化
- 风险管理
- 市场制造

**推荐系统**
- 个性化推荐
- 多目标优化
- 长期用户价值
- 探索与利用

### 工具和框架

**RL库**
- Stable Baselines3
- Ray RLlib
- OpenAI Baselines
- TensorFlow Agents

**环境**
- OpenAI Gym
- MuJoCo
- Unity ML-Agents
- PettingZoo（多智能体）

**可视化工具**
- TensorBoard
- Weights & Biases
- Matplotlib
- Plotly

## 总结

强化学习是AI领域最具挑战性和前景的分支之一。通过本模块的学习，你应该掌握：

**核心理论**：
- MDP的数学框架
- Bellman方程和最优性
- 各种学习算法的原理
- 深度强化学习的关键技术

**实践技能**：
- 算法实现和调试
- 环境交互和数据处理
- 超参数调优
- 性能评估和分析

**应用能力**：
- 问题建模和算法选择
- 系统集成和部署
- 安全性和鲁棒性考虑
- 跨领域应用

**未来方向**：
- 关注前沿研究进展
- 探索新的应用领域
- 参与开源项目
- 解决实际问题

强化学习正在快速发展，从游戏AI到机器人控制，从推荐系统到自动驾驶，都有广泛的应用前景。掌握这些核心技术将为你在AI领域的发展提供强大的工具。