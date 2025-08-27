# 模块七：智能体系统与多智能体协作

## 课程信息
- **模块编号**：Module 07
- **模块名称**：智能体系统与多智能体协作
- **学习目标**：掌握智能体架构、多智能体系统设计和协作机制
- **学时安排**：理论课程 14学时，实践课程 12学时

## 第一章：智能体基础理论

### 1.1 智能体概念与特征

#### 智能体定义

**基本概念**
```
智能体（Agent）是能够：
1. 感知环境（Perception）
2. 自主决策（Autonomous Decision Making）
3. 执行动作（Action Execution）
4. 适应变化（Adaptation）
的计算实体
```

**核心特征**

*自主性（Autonomy）*
- 独立运行能力
- 无需人工干预
- 自我管理和控制

*反应性（Reactivity）*
- 环境感知能力
- 及时响应变化
- 适应性行为

*主动性（Proactiveness）*
- 目标导向行为
- 主动采取行动
- 预测和规划

*社会性（Social Ability）*
- 与其他智能体交互
- 协作和竞争
- 通信和协调

#### 智能体分类

**按复杂度分类**

*简单反射智能体*
```
function SIMPLE-REFLEX-AGENT(percept) returns action
    static: rules, a set of condition-action rules
    
    state ← INTERPRET-INPUT(percept)
    rule ← RULE-MATCH(state, rules)
    action ← rule.ACTION
    return action
```

*基于模型的反射智能体*
```
function MODEL-BASED-REFLEX-AGENT(percept) returns action
    static: state, model, rules, action
    
    state ← UPDATE-STATE(state, action, percept, model)
    rule ← RULE-MATCH(state, rules)
    action ← rule.ACTION
    return action
```

*基于目标的智能体*
```
function GOAL-BASED-AGENT(percept) returns action
    static: state, model, goal, action
    
    state ← UPDATE-STATE(state, action, percept, model)
    action ← CHOOSE-ACTION(state, goal, model)
    return action
```

*基于效用的智能体*
```
function UTILITY-BASED-AGENT(percept) returns action
    static: state, model, utility, action
    
    state ← UPDATE-STATE(state, action, percept, model)
    action ← argmax_a EXPECTED-UTILITY(a, state, model, utility)
    return action
```

**按应用领域分类**

*软件智能体*
- 信息检索智能体
- 个人助理智能体
- 电子商务智能体
- 网络管理智能体

*机器人智能体*
- 移动机器人
- 工业机器人
- 服务机器人
- 探索机器人

*混合智能体*
- 人机协作系统
- 增强现实智能体
- 物联网智能体

### 1.2 智能体架构

#### 分层架构

**三层架构**

*反应层（Reactive Layer）*
```
功能：
- 快速响应紧急情况
- 基本行为模式
- 实时控制

特点：
- 低延迟
- 简单规则
- 直接映射
```

*执行层（Executive Layer）*
```
功能：
- 任务规划和调度
- 资源管理
- 行为协调

特点：
- 中等复杂度
- 局部优化
- 动态调整
```

*深思层（Deliberative Layer）*
```
功能：
- 长期规划
- 学习和适应
- 知识推理

特点：
- 高复杂度
- 全局优化
- 战略决策
```

**BDI架构（Belief-Desire-Intention）**

*信念（Beliefs）*
```
定义：智能体对环境状态的认知

表示：
- 逻辑公式
- 概率分布
- 知识图谱

更新：
- 感知输入
- 推理机制
- 不确定性处理
```

*愿望（Desires）*
```
定义：智能体希望达到的目标状态

类型：
- 成就目标（Achievement Goals）
- 维持目标（Maintenance Goals）
- 查询目标（Query Goals）

冲突处理：
- 优先级排序
- 效用评估
- 妥协机制
```

*意图（Intentions）*
```
定义：智能体承诺执行的行动计划

特征：
- 承诺性（Commitment）
- 持久性（Persistence）
- 可撤销性（Revisability）

执行：
- 计划生成
- 执行监控
- 动态调整
```

*BDI推理循环*
```
function BDI-AGENT-CYCLE()
    while true do
        // 感知环境
        percepts ← GET-PERCEPTS()
        
        // 更新信念
        beliefs ← UPDATE-BELIEFS(beliefs, percepts)
        
        // 生成选项
        options ← GENERATE-OPTIONS(beliefs, desires)
        
        // 过滤选项
        options ← FILTER-OPTIONS(options, beliefs, intentions)
        
        // 选择意图
        intentions ← SELECT-INTENTIONS(options, beliefs, desires, intentions)
        
        // 执行行动
        action ← EXECUTE(intentions)
        
        // 检查成功
        if SUCCEEDED(intentions, beliefs) then
            intentions ← intentions - {completed intentions}
        
        // 检查失败
        if IMPOSSIBLE(intentions, beliefs) then
            intentions ← intentions - {impossible intentions}
```

#### 认知架构

**ACT-R（Adaptive Control of Thought-Rational）**

*模块化设计*
```
感知模块：
- 视觉模块
- 听觉模块
- 触觉模块

认知模块：
- 目标模块
- 检索模块
- 想象模块

运动模块：
- 手动模块
- 语音模块
```

*知识表示*
```
程序性知识（产生式规则）：
IF goal is to add X and Y
   AND X is in focus
   AND Y is in focus
THEN set result to X + Y
   AND pop goal

陈述性知识（块结构）：
chunk-type: arithmetic-fact
   arg1: number
   arg2: number
   result: number

fact1:
   isa: arithmetic-fact
   arg1: 2
   arg2: 3
   result: 5
```

**SOAR（State, Operator, And Result）**

*问题空间*
```
状态（State）：当前情况描述
操作符（Operator）：可执行的动作
结果（Result）：动作执行后的新状态

搜索过程：
1. 状态详细化
2. 操作符选择
3. 操作符应用
4. 结果评估
```

*学习机制*
```
分块学习（Chunking）：
- 从问题解决经验中学习
- 生成新的产生式规则
- 提高问题解决效率

强化学习：
- 基于奖励信号
- 调整操作符偏好
- 优化决策策略
```

### 1.3 智能体通信

#### 通信语言

**KQML（Knowledge Query and Manipulation Language）**

*消息结构*
```
(performative
  :sender agent1
  :receiver agent2
  :content "(price IBM 100)"
  :language KIF
  :ontology NYSE-ticks)
```

*性能动词（Performatives）*
```
断言类：
- tell：告知信息
- deny：否认信息
- untell：撤回信息

查询类：
- ask-if：询问真假
- ask-one：询问一个答案
- ask-all：询问所有答案

响应类：
- reply：回复查询
- sorry：无法回答
```

**FIPA-ACL（Foundation for Intelligent Physical Agents - Agent Communication Language）**

*消息结构*
```
(
  (action
    (agent-identifier :name j)
    (sell
      :receiver (agent-identifier :name i)
      :content "(price (bid good02) 150)"
      :language fipa-sl
      :ontology hpl-auction
    )
  )
)
```

*通信行为*
```
承诺类：
- inform：通知信息
- confirm：确认信息
- agree：同意请求

指令类：
- request：请求行动
- query-if：查询条件
- propose：提出建议

声明类：
- accept-proposal：接受提议
- reject-proposal：拒绝提议
- cancel：取消请求
```

#### 协调机制

**合同网协议（Contract Net Protocol）**

*协议流程*
```
1. 任务公告（Task Announcement）
   管理者 → 所有承包者：任务描述

2. 投标（Bidding）
   承包者 → 管理者：投标书

3. 评估和授予（Evaluation and Award）
   管理者：评估投标，选择最佳承包者
   管理者 → 获胜承包者：合同授予
   管理者 → 其他承包者：拒绝通知

4. 执行和报告（Execution and Reporting）
   承包者：执行任务
   承包者 → 管理者：进度报告
```

*算法实现*
```python
class ContractNetManager:
    def __init__(self):
        self.contractors = []
        self.tasks = []
    
    def announce_task(self, task):
        """发布任务公告"""
        for contractor in self.contractors:
            contractor.receive_announcement(task)
    
    def collect_bids(self, task, timeout):
        """收集投标"""
        bids = []
        for contractor in self.contractors:
            bid = contractor.submit_bid(task, timeout)
            if bid:
                bids.append(bid)
        return bids
    
    def evaluate_bids(self, bids):
        """评估投标"""
        return min(bids, key=lambda x: x.cost)
    
    def award_contract(self, winning_bid):
        """授予合同"""
        winning_bid.contractor.accept_contract(winning_bid.task)
        
        # 通知其他承包者
        for contractor in self.contractors:
            if contractor != winning_bid.contractor:
                contractor.reject_notification(winning_bid.task)

class ContractNetContractor:
    def __init__(self, agent_id, capabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.current_tasks = []
    
    def receive_announcement(self, task):
        """接收任务公告"""
        if self.can_handle(task):
            return self.prepare_bid(task)
        return None
    
    def can_handle(self, task):
        """检查是否能处理任务"""
        return task.required_capability in self.capabilities
    
    def prepare_bid(self, task):
        """准备投标"""
        cost = self.estimate_cost(task)
        time = self.estimate_time(task)
        return Bid(self, task, cost, time)
    
    def estimate_cost(self, task):
        """估算成本"""
        base_cost = task.complexity * 10
        load_factor = len(self.current_tasks) * 0.1
        return base_cost * (1 + load_factor)
```

**拍卖机制**

*英式拍卖（English Auction）*
```
特点：
- 价格递增
- 公开竞价
- 最高价获胜

算法：
function ENGLISH_AUCTION(item, bidders, reserve_price):
    current_price = reserve_price
    active_bidders = bidders
    
    while len(active_bidders) > 1:
        # 征集更高出价
        new_bids = []
        for bidder in active_bidders:
            bid = bidder.make_bid(current_price)
            if bid > current_price:
                new_bids.append((bidder, bid))
        
        if not new_bids:
            break
        
        # 选择最高出价
        winner, highest_bid = max(new_bids, key=lambda x: x[1])
        current_price = highest_bid
        
        # 移除退出的竞标者
        active_bidders = [b for b, _ in new_bids]
    
    return active_bidders[0], current_price
```

*荷兰式拍卖（Dutch Auction）*
```
特点：
- 价格递减
- 首个接受者获胜
- 快速成交

算法：
function DUTCH_AUCTION(item, bidders, start_price, decrement):
    current_price = start_price
    
    while current_price > 0:
        for bidder in bidders:
            if bidder.willing_to_pay(current_price):
                return bidder, current_price
        
        current_price -= decrement
        time.sleep(auction_interval)
    
    return None, 0  # 流拍
```

*密封投标拍卖（Sealed-Bid Auction）*
```
第一价格密封投标：
- 所有竞标者同时提交密封投标
- 最高投标者获胜
- 支付自己的投标价格

第二价格密封投标（Vickrey拍卖）：
- 最高投标者获胜
- 支付第二高的投标价格
- 激励真实报价

function VICKREY_AUCTION(item, sealed_bids):
    sorted_bids = sort(sealed_bids, reverse=True)
    winner = sorted_bids[0].bidder
    price = sorted_bids[1].amount if len(sorted_bids) > 1 else 0
    return winner, price
```

## 第二章：多智能体系统

### 2.1 系统架构与组织

#### 组织结构

**层次结构（Hierarchical Structure）**

*特点*
```
优势：
- 清晰的指挥链
- 高效的决策传递
- 易于管理和控制

劣势：
- 单点故障风险
- 通信瓶颈
- 缺乏灵活性
```

*实现*
```python
class HierarchicalAgent:
    def __init__(self, agent_id, level, parent=None):
        self.agent_id = agent_id
        self.level = level
        self.parent = parent
        self.children = []
        self.tasks = []
    
    def add_child(self, child_agent):
        """添加子智能体"""
        self.children.append(child_agent)
        child_agent.parent = self
    
    def delegate_task(self, task):
        """任务委派"""
        if self.can_handle_directly(task):
            self.execute_task(task)
        else:
            # 分解任务并委派给子智能体
            subtasks = self.decompose_task(task)
            for subtask in subtasks:
                best_child = self.select_child_for_task(subtask)
                best_child.delegate_task(subtask)
    
    def report_to_parent(self, result):
        """向上级报告"""
        if self.parent:
            self.parent.receive_report(self, result)
    
    def coordinate_children(self):
        """协调子智能体"""
        for child in self.children:
            status = child.get_status()
            if status.needs_help:
                self.provide_assistance(child, status.problem)
```

**网络结构（Network Structure）**

*小世界网络*
```
特征：
- 高聚类系数
- 短平均路径长度
- 局部连接 + 少量长距离连接

优势：
- 高效信息传播
- 鲁棒性强
- 适应性好
```

*无标度网络*
```
特征：
- 幂律度分布
- 少数高度连接的枢纽节点
- 大量低度连接的普通节点

优势：
- 容错性强
- 信息传播效率高
- 自组织特性
```

**联邦结构（Federated Structure）**

*特点*
```
组织方式：
- 自治域（Autonomous Domains）
- 域间协调机制
- 联邦协议

优势：
- 保持局部自治
- 支持异构系统
- 可扩展性好
```

#### 协调策略

**集中式协调**

*中央协调器*
```python
class CentralCoordinator:
    def __init__(self):
        self.agents = []
        self.global_state = {}
        self.task_queue = []
    
    def register_agent(self, agent):
        """注册智能体"""
        self.agents.append(agent)
        agent.set_coordinator(self)
    
    def coordinate_task_allocation(self, tasks):
        """协调任务分配"""
        allocation = self.optimize_allocation(tasks, self.agents)
        
        for agent, assigned_tasks in allocation.items():
            agent.receive_tasks(assigned_tasks)
    
    def optimize_allocation(self, tasks, agents):
        """优化任务分配"""
        # 使用匈牙利算法或其他优化方法
        cost_matrix = self.build_cost_matrix(tasks, agents)
        assignment = hungarian_algorithm(cost_matrix)
        return self.convert_to_allocation(assignment, tasks, agents)
    
    def handle_conflicts(self, conflicts):
        """处理冲突"""
        for conflict in conflicts:
            resolution = self.resolve_conflict(conflict)
            self.broadcast_resolution(resolution)
```

**分布式协调**

*共识算法*
```python
class ConsensusAgent:
    def __init__(self, agent_id, initial_value):
        self.agent_id = agent_id
        self.value = initial_value
        self.neighbors = []
        self.round = 0
    
    def add_neighbor(self, neighbor):
        """添加邻居"""
        self.neighbors.append(neighbor)
    
    def consensus_round(self):
        """共识轮次"""
        # 收集邻居的值
        neighbor_values = [neighbor.get_value() for neighbor in self.neighbors]
        all_values = [self.value] + neighbor_values
        
        # 更新自己的值（平均值共识）
        self.value = sum(all_values) / len(all_values)
        self.round += 1
    
    def has_converged(self, tolerance=1e-6):
        """检查是否收敛"""
        neighbor_values = [neighbor.get_value() for neighbor in self.neighbors]
        return all(abs(self.value - nv) < tolerance for nv in neighbor_values)
```

*分布式任务分配*
```python
class DistributedTaskAllocation:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.local_tasks = []
        self.capabilities = set()
        self.load = 0
    
    def announce_capability(self, capability):
        """宣告能力"""
        self.capabilities.add(capability)
        self.broadcast_to_neighbors({
            'type': 'capability_announcement',
            'agent': self.agent_id,
            'capability': capability
        })
    
    def receive_task_request(self, task, requester):
        """接收任务请求"""
        if self.can_handle(task) and self.load < self.max_load:
            bid = self.calculate_bid(task)
            self.send_message(requester, {
                'type': 'task_bid',
                'task_id': task.id,
                'bid': bid,
                'agent': self.agent_id
            })
    
    def distributed_allocation(self, new_task):
        """分布式任务分配"""
        if self.can_handle(new_task):
            self.local_tasks.append(new_task)
        else:
            # 寻找合适的智能体
            candidates = self.find_capable_agents(new_task)
            if candidates:
                best_agent = min(candidates, key=lambda a: a.load)
                self.forward_task(new_task, best_agent)
            else:
                # 任务分解
                subtasks = self.decompose_task(new_task)
                for subtask in subtasks:
                    self.distributed_allocation(subtask)
```

### 2.2 协作与竞争

#### 协作机制

**任务分解与分配**

*分解策略*
```python
class TaskDecomposition:
    def __init__(self):
        self.decomposition_rules = {}
    
    def decompose_task(self, task):
        """任务分解"""
        if task.type in self.decomposition_rules:
            return self.decomposition_rules[task.type](task)
        else:
            return self.generic_decomposition(task)
    
    def generic_decomposition(self, task):
        """通用分解方法"""
        subtasks = []
        
        # 基于依赖关系分解
        dependencies = task.get_dependencies()
        for dep in dependencies:
            if dep.can_be_parallelized():
                subtasks.extend(self.parallel_decomposition(dep))
            else:
                subtasks.append(dep)
        
        return subtasks
    
    def parallel_decomposition(self, task):
        """并行分解"""
        num_parts = min(task.complexity, self.available_agents())
        return [task.create_subtask(i, num_parts) for i in range(num_parts)]
```

*分配算法*
```python
def hungarian_assignment(cost_matrix):
    """匈牙利算法求解任务分配"""
    from scipy.optimize import linear_sum_assignment
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_indices, col_indices].sum()
    
    assignment = list(zip(row_indices, col_indices))
    return assignment, total_cost

def auction_algorithm(agents, tasks, epsilon=0.1):
    """拍卖算法"""
    prices = {task.id: 0 for task in tasks}
    assignment = {}
    
    while len(assignment) < len(agents):
        for agent in agents:
            if agent.id not in assignment:
                # 计算每个任务的净效用
                utilities = {}
                for task in tasks:
                    if task.id not in assignment.values():
                        utility = agent.get_utility(task) - prices[task.id]
                        utilities[task.id] = utility
                
                # 选择最佳和次佳任务
                sorted_tasks = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
                best_task = sorted_tasks[0][0]
                best_utility = sorted_tasks[0][1]
                
                if len(sorted_tasks) > 1:
                    second_best_utility = sorted_tasks[1][1]
                else:
                    second_best_utility = 0
                
                # 更新价格和分配
                price_increment = best_utility - second_best_utility + epsilon
                prices[best_task] += price_increment
                assignment[agent.id] = best_task
    
    return assignment
```

**信息共享**

*黑板系统*
```python
class Blackboard:
    def __init__(self):
        self.knowledge_sources = []
        self.data_structures = {}
        self.control_strategy = None
    
    def add_knowledge_source(self, ks):
        """添加知识源"""
        self.knowledge_sources.append(ks)
        ks.set_blackboard(self)
    
    def write_data(self, key, value, author):
        """写入数据"""
        self.data_structures[key] = {
            'value': value,
            'author': author,
            'timestamp': time.time()
        }
        
        # 通知相关知识源
        self.notify_knowledge_sources(key, value)
    
    def read_data(self, key):
        """读取数据"""
        return self.data_structures.get(key)
    
    def notify_knowledge_sources(self, key, value):
        """通知知识源"""
        for ks in self.knowledge_sources:
            if ks.is_interested_in(key):
                ks.process_update(key, value)

class KnowledgeSource:
    def __init__(self, name, expertise):
        self.name = name
        self.expertise = expertise
        self.blackboard = None
    
    def set_blackboard(self, blackboard):
        self.blackboard = blackboard
    
    def is_interested_in(self, key):
        """检查是否对某个数据感兴趣"""
        return any(exp in key for exp in self.expertise)
    
    def process_update(self, key, value):
        """处理数据更新"""
        result = self.analyze_data(key, value)
        if result:
            self.blackboard.write_data(f"{self.name}_analysis", result, self.name)
```

#### 竞争机制

**资源竞争**

*资源分配算法*
```python
class ResourceManager:
    def __init__(self):
        self.resources = {}
        self.allocations = {}
        self.waiting_queue = []
    
    def request_resource(self, agent_id, resource_type, amount, priority=1):
        """资源请求"""
        request = ResourceRequest(agent_id, resource_type, amount, priority)
        
        if self.can_allocate(request):
            self.allocate_resource(request)
            return True
        else:
            self.waiting_queue.append(request)
            return False
    
    def can_allocate(self, request):
        """检查是否可以分配"""
        available = self.resources.get(request.resource_type, 0)
        return available >= request.amount
    
    def allocate_resource(self, request):
        """分配资源"""
        self.resources[request.resource_type] -= request.amount
        
        if request.agent_id not in self.allocations:
            self.allocations[request.agent_id] = {}
        
        current = self.allocations[request.agent_id].get(request.resource_type, 0)
        self.allocations[request.agent_id][request.resource_type] = current + request.amount
    
    def release_resource(self, agent_id, resource_type, amount):
        """释放资源"""
        self.resources[resource_type] += amount
        self.allocations[agent_id][resource_type] -= amount
        
        # 处理等待队列
        self.process_waiting_queue()
    
    def process_waiting_queue(self):
        """处理等待队列"""
        # 按优先级排序
        self.waiting_queue.sort(key=lambda x: x.priority, reverse=True)
        
        allocated = []
        for i, request in enumerate(self.waiting_queue):
            if self.can_allocate(request):
                self.allocate_resource(request)
                allocated.append(i)
        
        # 移除已分配的请求
        for i in reversed(allocated):
            del self.waiting_queue[i]
```

**市场机制**

*双边拍卖*
```python
class DoubleAuction:
    def __init__(self):
        self.buy_orders = []  # (price, quantity, buyer_id)
        self.sell_orders = []  # (price, quantity, seller_id)
        self.transactions = []
    
    def submit_buy_order(self, buyer_id, price, quantity):
        """提交买单"""
        order = BuyOrder(buyer_id, price, quantity)
        self.buy_orders.append(order)
        self.match_orders()
    
    def submit_sell_order(self, seller_id, price, quantity):
        """提交卖单"""
        order = SellOrder(seller_id, price, quantity)
        self.sell_orders.append(order)
        self.match_orders()
    
    def match_orders(self):
        """订单匹配"""
        # 按价格排序
        self.buy_orders.sort(key=lambda x: x.price, reverse=True)  # 高价优先
        self.sell_orders.sort(key=lambda x: x.price)  # 低价优先
        
        i, j = 0, 0
        while i < len(self.buy_orders) and j < len(self.sell_orders):
            buy_order = self.buy_orders[i]
            sell_order = self.sell_orders[j]
            
            if buy_order.price >= sell_order.price:
                # 可以成交
                transaction_price = (buy_order.price + sell_order.price) / 2
                transaction_quantity = min(buy_order.quantity, sell_order.quantity)
                
                transaction = Transaction(
                    buy_order.buyer_id,
                    sell_order.seller_id,
                    transaction_price,
                    transaction_quantity
                )
                self.transactions.append(transaction)
                
                # 更新订单数量
                buy_order.quantity -= transaction_quantity
                sell_order.quantity -= transaction_quantity
                
                # 移除完成的订单
                if buy_order.quantity == 0:
                    i += 1
                if sell_order.quantity == 0:
                    j += 1
            else:
                break
        
        # 清理完成的订单
        self.buy_orders = [order for order in self.buy_orders if order.quantity > 0]
        self.sell_orders = [order for order in self.sell_orders if order.quantity > 0]
```

### 2.3 学习与适应

#### 多智能体强化学习

**独立学习**

*Q-Learning*
```python
class IndependentQLearning:
    def __init__(self, agent_id, state_space, action_space, learning_rate=0.1, discount=0.9):
        self.agent_id = agent_id
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = 0.1
    
    def get_q_value(self, state, action):
        """获取Q值"""
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state, available_actions):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_actions):
        """更新Q值"""
        current_q = self.get_q_value(state, action)
        
        if next_actions:
            max_next_q = max(self.get_q_value(next_state, a) for a in next_actions)
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
```

**联合动作学习**

*Multi-Agent Q-Learning*
```python
class MultiAgentQLearning:
    def __init__(self, agents, state_space, joint_action_space):
        self.agents = agents
        self.q_table = {}  # (state, joint_action) -> q_value
        self.learning_rate = 0.1
        self.discount = 0.9
    
    def get_joint_q_value(self, state, joint_action):
        """获取联合Q值"""
        return self.q_table.get((state, tuple(joint_action)), 0.0)
    
    def choose_joint_action(self, state, available_joint_actions):
        """选择联合动作"""
        if random.random() < 0.1:  # epsilon-greedy
            return random.choice(available_joint_actions)
        else:
            q_values = [self.get_joint_q_value(state, ja) for ja in available_joint_actions]
            max_q = max(q_values)
            best_actions = [ja for ja, q in zip(available_joint_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_joint_q_value(self, state, joint_action, joint_reward, next_state, next_joint_actions):
        """更新联合Q值"""
        current_q = self.get_joint_q_value(state, joint_action)
        
        if next_joint_actions:
            max_next_q = max(self.get_joint_q_value(next_state, ja) for ja in next_joint_actions)
        else:
            max_next_q = 0
        
        # 使用联合奖励的平均值
        avg_reward = sum(joint_reward) / len(joint_reward)
        new_q = current_q + self.learning_rate * (avg_reward + self.discount * max_next_q - current_q)
        self.q_table[(state, tuple(joint_action))] = new_q
```

**策略梯度方法**

*Multi-Agent Policy Gradient*
```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class MultiAgentPolicyGradient:
    def __init__(self, agents, state_dim, action_dims):
        self.agents = agents
        self.policies = {}
        self.optimizers = {}
        
        for agent_id, action_dim in zip(agents, action_dims):
            self.policies[agent_id] = PolicyNetwork(state_dim, action_dim)
            self.optimizers[agent_id] = optim.Adam(self.policies[agent_id].parameters(), lr=0.01)
    
    def select_actions(self, state):
        """选择动作"""
        actions = {}
        log_probs = {}
        
        state_tensor = torch.FloatTensor(state)
        
        for agent_id in self.agents:
            action_probs = self.policies[agent_id](state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            actions[agent_id] = action.item()
            log_probs[agent_id] = action_dist.log_prob(action)
        
        return actions, log_probs
    
    def update_policies(self, trajectories):
        """更新策略"""
        for agent_id in self.agents:
            policy_loss = 0
            
            for trajectory in trajectories:
                returns = self.calculate_returns(trajectory['rewards'][agent_id])
                
                for log_prob, return_val in zip(trajectory['log_probs'][agent_id], returns):
                    policy_loss -= log_prob * return_val
            
            self.optimizers[agent_id].zero_grad()
            policy_loss.backward()
            self.optimizers[agent_id].step()
    
    def calculate_returns(self, rewards, gamma=0.99):
        """计算回报"""
        returns = []
        R = 0
        
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        
        return returns
```

#### 进化算法

**遗传算法**

```python
import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, gene_length, mutation_rate=0.01, crossover_rate=0.8):
        self.population_size = population_size
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """初始化种群"""
        return [self.random_individual() for _ in range(self.population_size)]
    
    def random_individual(self):
        """生成随机个体"""
        return [random.randint(0, 1) for _ in range(self.gene_length)]
    
    def fitness(self, individual):
        """适应度函数（需要根据具体问题实现）"""
        return sum(individual)  # 简单示例
    
    def selection(self):
        """选择操作"""
        # 轮盘赌选择
        fitness_scores = [self.fitness(ind) for ind in self.population]
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        probabilities = [f / total_fitness for f in fitness_scores]
        return np.random.choice(self.population, p=probabilities)
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        crossover_point = random.randint(1, self.gene_length - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutation(self, individual):
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # 翻转位
        
        return mutated
    
    def evolve_generation(self):
        """进化一代"""
        new_population = []
        
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
    
    def run(self, generations):
        """运行遗传算法"""
        for generation in range(generations):
            self.evolve_generation()
            
            # 输出最佳个体
            best_individual = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_individual)
            print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        return max(self.population, key=self.fitness)
```

## 实践项目

### 项目一：分布式任务调度系统

```python
class DistributedTaskScheduler:
    def __init__(self, agent_id, neighbors):
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.local_tasks = []
        self.capabilities = set()
        self.load = 0
        self.max_load = 10
    
    def receive_task(self, task):
        """接收新任务"""
        if self.can_handle_locally(task):
            self.schedule_locally(task)
        else:
            self.negotiate_task_allocation(task)
    
    def can_handle_locally(self, task):
        """检查是否可以本地处理"""
        return (task.required_capability in self.capabilities and 
                self.load + task.workload <= self.max_load)
    
    def schedule_locally(self, task):
        """本地调度任务"""
        self.local_tasks.append(task)
        self.load += task.workload
        print(f"Agent {self.agent_id} scheduled task {task.id} locally")
    
    def negotiate_task_allocation(self, task):
        """协商任务分配"""
        # 向邻居发送任务请求
        bids = []
        for neighbor in self.neighbors:
            bid = neighbor.request_bid(task)
            if bid:
                bids.append(bid)
        
        if bids:
            # 选择最佳投标
            best_bid = min(bids, key=lambda b: b.cost)
            best_bid.agent.accept_task(task)
            print(f"Task {task.id} allocated to Agent {best_bid.agent.agent_id}")
        else:
            # 任务分解
            subtasks = self.decompose_task(task)
            for subtask in subtasks:
                self.receive_task(subtask)
    
    def request_bid(self, task):
        """请求投标"""
        if self.can_handle_locally(task):
            cost = self.calculate_cost(task)
            return Bid(self, task, cost)
        return None
    
    def calculate_cost(self, task):
        """计算任务成本"""
        base_cost = task.workload
        load_penalty = self.load * 0.1
        return base_cost + load_penalty

class Task:
    def __init__(self, task_id, required_capability, workload, deadline):
        self.id = task_id
        self.required_capability = required_capability
        self.workload = workload
        self.deadline = deadline

class Bid:
    def __init__(self, agent, task, cost):
        self.agent = agent
        self.task = task
        self.cost = cost

# 使用示例
agents = [DistributedTaskScheduler(i, []) for i in range(5)]

# 建立邻居关系
for i, agent in enumerate(agents):
    agent.neighbors = [agents[j] for j in range(len(agents)) if j != i]
    agent.capabilities = {f"capability_{i%3}"}  # 分配不同能力

# 创建任务
tasks = [
    Task(1, "capability_0", 3, 10),
    Task(2, "capability_1", 5, 15),
    Task(3, "capability_2", 2, 8)
]

# 分配任务
for task in tasks:
    agents[0].receive_task(task)
```

### 项目二：多智能体路径规划

```python
import numpy as np
from collections import deque

class MultiAgentPathPlanning:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.obstacles = obstacles
        self.agents = {}
        
        # 设置障碍物
        for obs in obstacles:
            self.grid[obs[0], obs[1]] = -1
    
    def add_agent(self, agent_id, start_pos, goal_pos):
        """添加智能体"""
        self.agents[agent_id] = {
            'position': start_pos,
            'goal': goal_pos,
            'path': [],
            'status': 'planning'
        }
    
    def a_star_search(self, start, goal, agent_id, time_step=0):
        """A*搜索算法（考虑时空冲突）"""
        from heapq import heappush, heappop
        
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        def is_valid_move(pos, t):
            x, y = pos
            if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
                return False
            if self.grid[x, y] == -1:  # 障碍物
                return False
            
            # 检查与其他智能体的冲突
            for other_id, other_agent in self.agents.items():
                if other_id != agent_id and len(other_agent['path']) > t:
                    if other_agent['path'][t] == pos:
                        return False
            
            return True
        
        open_set = [(heuristic(start), 0, start, [start])]
        closed_set = set()
        
        while open_set:
            f_score, g_score, current, path = heappop(open_set)
            
            if current == goal:
                return path
            
            if (current, g_score) in closed_set:
                continue
            
            closed_set.add((current, g_score))
            
            # 探索邻居
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]:  # 包括等待
                next_pos = (current[0] + dx, current[1] + dy)
                next_time = g_score + 1
                
                if is_valid_move(next_pos, next_time):
                    new_path = path + [next_pos]
                    f_score = next_time + heuristic(next_pos)
                    heappush(open_set, (f_score, next_time, next_pos, new_path))
        
        return None  # 无解
    
    def cooperative_planning(self):
        """协作路径规划"""
        # 按优先级排序（可以基于距离、重要性等）
        agent_priorities = sorted(self.agents.keys(), 
                                key=lambda aid: self.manhattan_distance(
                                    self.agents[aid]['position'], 
                                    self.agents[aid]['goal']
                                ))
        
        for agent_id in agent_priorities:
            agent = self.agents[agent_id]
            path = self.a_star_search(agent['position'], agent['goal'], agent_id)
            
            if path:
                agent['path'] = path
                agent['status'] = 'planned'
                print(f"Agent {agent_id} path: {path}")
            else:
                agent['status'] = 'failed'
                print(f"Agent {agent_id} failed to find path")
    
    def manhattan_distance(self, pos1, pos2):
        """曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def execute_step(self, time_step):
        """执行一个时间步"""
        for agent_id, agent in self.agents.items():
            if agent['status'] == 'planned' and len(agent['path']) > time_step:
                agent['position'] = agent['path'][time_step]
                
                if agent['position'] == agent['goal']:
                    agent['status'] = 'completed'
                    print(f"Agent {agent_id} reached goal")
    
    def visualize(self, time_step=0):
        """可视化当前状态"""
        vis_grid = self.grid.copy()
        
        for agent_id, agent in self.agents.items():
            if agent['status'] in ['planned', 'completed']:
                if len(agent['path']) > time_step:
                    pos = agent['path'][time_step]
                else:
                    pos = agent['goal']
                vis_grid[pos[0], pos[1]] = agent_id + 1
        
        print(f"Time step {time_step}:")
        print(vis_grid)
        print()

# 使用示例
planner = MultiAgentPathPlanning((8, 8), [(2, 2), (3, 3), (5, 5)])

# 添加智能体
planner.add_agent(0, (0, 0), (7, 7))
planner.add_agent(1, (0, 7), (7, 0))
planner.add_agent(2, (7, 0), (0, 7))

# 规划路径
planner.cooperative_planning()

# 模拟执行
for t in range(15):
    planner.visualize(t)
    planner.execute_step(t)
```

### 项目三：智能体协商系统

```python
class NegotiationAgent:
    def __init__(self, agent_id, preferences, reservation_value):
        self.agent_id = agent_id
        self.preferences = preferences  # 对不同议题的偏好
        self.reservation_value = reservation_value  # 保留价值
        self.negotiation_history = []
        self.strategy = 'tit_for_tat'
    
    def generate_offer(self, round_num, opponent_last_offer=None):
        """生成报价"""
        if self.strategy == 'concession':
            return self.concession_strategy(round_num)
        elif self.strategy == 'tit_for_tat':
            return self.tit_for_tat_strategy(opponent_last_offer)
        else:
            return self.random_strategy()
    
    def concession_strategy(self, round_num, max_rounds=10):
        """让步策略"""
        concession_rate = round_num / max_rounds
        
        offer = {}
        for issue, (min_val, max_val) in self.preferences.items():
            # 从最优值向保留值让步
            current_val = max_val - concession_rate * (max_val - min_val)
            offer[issue] = max(current_val, min_val)
        
        return offer
    
    def tit_for_tat_strategy(self, opponent_last_offer):
        """针锋相对策略"""
        if opponent_last_offer is None:
            # 首轮合作性报价
            return self.generate_cooperative_offer()
        
        # 分析对手让步程度
        opponent_concession = self.analyze_concession(opponent_last_offer)
        
        # 相应让步
        offer = {}
        for issue, (min_val, max_val) in self.preferences.items():
            concession = opponent_concession * (max_val - min_val)
            offer[issue] = max_val - concession
        
        return offer
    
    def evaluate_offer(self, offer):
        """评估报价"""
        utility = 0
        for issue, value in offer.items():
            if issue in self.preferences:
                min_val, max_val = self.preferences[issue]
                normalized_value = (value - min_val) / (max_val - min_val)
                utility += normalized_value
        
        return utility / len(self.preferences)
    
    def accept_offer(self, offer):
        """决定是否接受报价"""
        utility = self.evaluate_offer(offer)
        return utility >= self.reservation_value
    
    def analyze_concession(self, offer):
        """分析对手让步程度"""
        if not self.negotiation_history:
            return 0
        
        last_offer = self.negotiation_history[-1]
        current_utility = self.evaluate_offer(offer)
        last_utility = self.evaluate_offer(last_offer)
        
        return max(0, current_utility - last_utility)
    
    def generate_cooperative_offer(self):
        """生成合作性报价"""
        offer = {}
        for issue, (min_val, max_val) in self.preferences.items():
            # 中等让步
            offer[issue] = min_val + 0.7 * (max_val - min_val)
        return offer

class NegotiationMediator:
    def __init__(self, agents, issues, max_rounds=20):
        self.agents = agents
        self.issues = issues
        self.max_rounds = max_rounds
        self.negotiation_log = []
        self.current_round = 0
    
    def conduct_negotiation(self):
        """进行协商"""
        print(f"Starting negotiation with {len(self.agents)} agents")
        
        while self.current_round < self.max_rounds:
            self.current_round += 1
            print(f"\nRound {self.current_round}:")
            
            round_offers = {}
            
            # 每个智能体生成报价
            for agent in self.agents:
                opponent_offers = [offer for aid, offer in round_offers.items() if aid != agent.agent_id]
                last_opponent_offer = opponent_offers[-1] if opponent_offers else None
                
                offer = agent.generate_offer(self.current_round, last_opponent_offer)
                round_offers[agent.agent_id] = offer
                
                print(f"Agent {agent.agent_id} offers: {offer}")
            
            # 检查是否有智能体接受其他智能体的报价
            agreements = self.check_agreements(round_offers)
            
            if agreements:
                print(f"Agreement reached: {agreements}")
                return agreements
            
            # 记录本轮协商
            self.negotiation_log.append(round_offers)
            
            # 更新智能体的协商历史
            for agent in self.agents:
                agent.negotiation_history.append(round_offers[agent.agent_id])
        
        print("Negotiation failed - maximum rounds reached")
        return None
    
    def check_agreements(self, round_offers):
        """检查是否达成协议"""
        agreements = []
        
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                offer1 = round_offers[agent1.agent_id]
                offer2 = round_offers[agent2.agent_id]
                
                if agent1.accept_offer(offer2) and agent2.accept_offer(offer1):
                    agreements.append((agent1.agent_id, agent2.agent_id, offer1, offer2))
        
        return agreements

# 使用示例
agent1 = NegotiationAgent(
    agent_id=1,
    preferences={'price': (100, 200), 'quality': (0.5, 1.0), 'delivery': (1, 7)},
    reservation_value=0.6
)

agent2 = NegotiationAgent(
    agent_id=2,
    preferences={'price': (80, 150), 'quality': (0.7, 1.0), 'delivery': (2, 5)},
    reservation_value=0.5
)

mediator = NegotiationMediator(
    agents=[agent1, agent2],
    issues=['price', 'quality', 'delivery']
)

result = mediator.conduct_negotiation()
```

### 项目四：群体智能优化

```python
import numpy as np
import random
import math

class ParticleSwarmOptimization:
    def __init__(self, num_particles, dimensions, bounds, objective_function):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.objective_function = objective_function
        
        # PSO参数
        self.w = 0.729  # 惯性权重
        self.c1 = 1.49445  # 个体学习因子
        self.c2 = 1.49445  # 社会学习因子
        
        # 初始化粒子群
        self.particles = self.initialize_particles()
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        self.update_global_best()
    
    def initialize_particles(self):
        """初始化粒子群"""
        particles = []
        
        for i in range(self.num_particles):
            position = np.random.uniform(
                [bound[0] for bound in self.bounds],
                [bound[1] for bound in self.bounds]
            )
            
            velocity = np.random.uniform(
                -abs(np.array([bound[1] - bound[0] for bound in self.bounds])),
                abs(np.array([bound[1] - bound[0] for bound in self.bounds]))
            )
            
            particle = {
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': self.objective_function(position)
            }
            
            particles.append(particle)
        
        return particles
    
    def update_global_best(self):
        """更新全局最优"""
        for particle in self.particles:
            if particle['best_fitness'] < self.global_best_fitness:
                self.global_best_fitness = particle['best_fitness']
                self.global_best_position = particle['best_position'].copy()
    
    def update_particle(self, particle):
        """更新粒子状态"""
        # 更新速度
        r1, r2 = random.random(), random.random()
        
        cognitive_component = self.c1 * r1 * (particle['best_position'] - particle['position'])
        social_component = self.c2 * r2 * (self.global_best_position - particle['position'])
        
        particle['velocity'] = (self.w * particle['velocity'] + 
                              cognitive_component + social_component)
        
        # 更新位置
        particle['position'] += particle['velocity']
        
        # 边界处理
        for i, (min_bound, max_bound) in enumerate(self.bounds):
            if particle['position'][i] < min_bound:
                particle['position'][i] = min_bound
                particle['velocity'][i] = 0
            elif particle['position'][i] > max_bound:
                particle['position'][i] = max_bound
                particle['velocity'][i] = 0
        
        # 更新个体最优
        fitness = self.objective_function(particle['position'])
        if fitness < particle['best_fitness']:
            particle['best_fitness'] = fitness
            particle['best_position'] = particle['position'].copy()
    
    def optimize(self, max_iterations):
        """执行优化"""
        for iteration in range(max_iterations):
            for particle in self.particles:
                self.update_particle(particle)
            
            self.update_global_best()
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness}")
        
        return self.global_best_position, self.global_best_fitness

class AntColonyOptimization:
    def __init__(self, num_ants, num_cities, distance_matrix, alpha=1, beta=2, evaporation_rate=0.5):
        self.num_ants = num_ants
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta   # 启发式信息重要程度
        self.evaporation_rate = evaporation_rate
        
        # 初始化信息素矩阵
        self.pheromone_matrix = np.ones((num_cities, num_cities))
        
        self.best_path = None
        self.best_distance = float('inf')
    
    def calculate_probability(self, current_city, unvisited_cities):
        """计算转移概率"""
        probabilities = []
        
        for city in unvisited_cities:
            pheromone = self.pheromone_matrix[current_city][city] ** self.alpha
            heuristic = (1.0 / self.distance_matrix[current_city][city]) ** self.beta
            probabilities.append(pheromone * heuristic)
        
        total = sum(probabilities)
        return [p / total for p in probabilities] if total > 0 else [1.0 / len(unvisited_cities)] * len(unvisited_cities)
    
    def construct_solution(self):
        """构造解"""
        start_city = random.randint(0, self.num_cities - 1)
        path = [start_city]
        unvisited = list(range(self.num_cities))
        unvisited.remove(start_city)
        
        current_city = start_city
        
        while unvisited:
            probabilities = self.calculate_probability(current_city, unvisited)
            
            # 轮盘赌选择
            rand = random.random()
            cumulative_prob = 0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    next_city = unvisited[i]
                    break
            else:
                next_city = unvisited[-1]
            
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return path
    
    def calculate_path_distance(self, path):
        """计算路径距离"""
        distance = 0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]  # 回到起点
            distance += self.distance_matrix[from_city][to_city]
        return distance
    
    def update_pheromones(self, ant_paths):
        """更新信息素"""
        # 信息素蒸发
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        
        # 信息素增强
        for path in ant_paths:
            distance = self.calculate_path_distance(path)
            pheromone_deposit = 1.0 / distance
            
            for i in range(len(path)):
                from_city = path[i]
                to_city = path[(i + 1) % len(path)]
                self.pheromone_matrix[from_city][to_city] += pheromone_deposit
                self.pheromone_matrix[to_city][from_city] += pheromone_deposit
    
    def optimize(self, max_iterations):
        """执行优化"""
        for iteration in range(max_iterations):
            ant_paths = []
            
            # 每只蚂蚁构造解
            for ant in range(self.num_ants):
                path = self.construct_solution()
                ant_paths.append(path)
                
                # 更新最优解
                distance = self.calculate_path_distance(path)
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # 更新信息素
            self.update_pheromones(ant_paths)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best distance = {self.best_distance}")
        
        return self.best_path, self.best_distance

# 使用示例

# PSO优化示例
def sphere_function(x):
    """球函数（测试函数）"""
    return sum(xi**2 for xi in x)

pso = ParticleSwarmOptimization(
    num_particles=30,
    dimensions=10,
    bounds=[(-5.12, 5.12)] * 10,
    objective_function=sphere_function
)

best_position, best_fitness = pso.optimize(100)
print(f"PSO Best solution: {best_position}")
print(f"PSO Best fitness: {best_fitness}")

# ACO优化示例（TSP问题）
np.random.seed(42)
num_cities = 10
cities = np.random.rand(num_cities, 2) * 100

# 计算距离矩阵
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distance_matrix[i][j] = np.sqrt(sum((cities[i] - cities[j])**2))

aco = AntColonyOptimization(
    num_ants=20,
    num_cities=num_cities,
    distance_matrix=distance_matrix
)

best_path, best_distance = aco.optimize(100)
print(f"ACO Best path: {best_path}")
print(f"ACO Best distance: {best_distance}")
```

## 学习评估

### 理论评估（40%）

**概念理解**
1. 智能体的基本特征和分类
2. BDI架构的核心组件
3. 多智能体系统的组织结构
4. 协调机制和通信协议
5. 学习与适应算法

**分析能力**
1. 智能体架构设计分析
2. 协作与竞争机制比较
3. 多智能体学习算法评估
4. 系统性能优化策略

### 实践评估（60%）

**编程实现**
1. 智能体基础架构实现
2. 通信协议设计与实现
3. 协调算法编程
4. 学习算法集成

**项目评估**
1. 分布式任务调度系统
2. 多智能体路径规划
3. 智能体协商系统
4. 群体智能优化

**系统集成**
1. 多模块集成能力
2. 系统稳定性和鲁棒性
3. 性能优化和扩展性
4. 实际应用场景适配

## 延伸学习

### 前沿研究方向

**深度多智能体强化学习**
- MADDPG（Multi-Agent Deep Deterministic Policy Gradient）
- QMIX（Monotonic Value Function Factorisation）
- COMA（Counterfactual Multi-Agent Policy Gradients）
- MAPPO（Multi-Agent Proximal Policy Optimization）

**大规模多智能体系统**
- 可扩展性架构设计
- 分层协调机制
- 动态组织结构
- 容错与恢复机制

**人机协作智能体**
- 混合智能系统
- 人机交互界面
- 信任与透明度
- 协作学习机制

### 应用领域

**智能交通系统**
- 自动驾驶车辆协调
- 交通流量优化
- 智能信号控制
- 路径规划与导航

**智能制造**
- 生产调度优化
- 设备协调控制
- 质量监控系统
- 供应链管理

**智能电网**
- 分布式能源管理
- 负载平衡控制
- 故障检测与恢复
- 需求响应优化

**金融科技**
- 算法交易系统
- 风险管理
- 欺诈检测
- 投资组合优化

### 工具和框架

**开发平台**
- JADE（Java Agent DEvelopment Framework）
- SPADE（Smart Python Agent Development Environment）
- Mesa（Agent-based modeling in Python）
- NetLogo（Multi-agent programmable modeling environment）

**仿真环境**
- SUMO（Simulation of Urban MObility）
- AirSim（Autonomous vehicle simulation）
- OpenAI Gym（Multi-agent environments）
- PettingZoo（Multi-agent reinforcement learning environments）

**深度学习框架**
- PyTorch（支持多智能体强化学习）
- TensorFlow（分布式训练支持）
- Ray RLlib（可扩展强化学习库）
- Stable Baselines3（强化学习算法实现）

## 总结

智能体系统与多智能体协作是人工智能领域的重要分支，它研究如何设计和实现能够自主决策、相互协作的智能系统。通过本模块的学习，学生将掌握：

1. **理论基础**：智能体的基本概念、架构设计和通信机制
2. **系统设计**：多智能体系统的组织结构和协调策略
3. **算法实现**：协作与竞争算法、学习与适应机制
4. **实践应用**：分布式系统、路径规划、协商系统等实际项目

这些知识和技能为学生在智能系统设计、分布式计算、机器人协作等领域的深入研究和实际应用奠定了坚实基础。随着物联网、边缘计算和人工智能技术的快速发展，多智能体系统将在更多领域发挥重要作用。