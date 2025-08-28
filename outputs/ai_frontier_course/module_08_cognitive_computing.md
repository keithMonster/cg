# 模块八：认知计算与上下文理解

## 课程信息
- **模块编号**：Module 08
- **模块名称**：认知计算与上下文理解
- **学习目标**：掌握认知计算原理、上下文建模和智能推理技术
- **学时安排**：理论课程 16学时，实践课程 14学时

## 第一章：认知计算基础

### 1.1 认知计算概念与原理

#### 认知计算定义

**什么是认知计算？**

认知计算就像是"让计算机学会像人一样思考"的技术。想象一下，如果计算机不仅能计算数字，还能像人类一样理解、学习、推理和判断，那就是认知计算要实现的目标。

**认知计算的五大"超能力"**：
1. **机器学习** - 像学生一样从经验中学习
2. **自然语言处理** - 像翻译官一样理解人类语言
3. **知识表示** - 像图书管理员一样整理和存储知识
4. **推理机制** - 像侦探一样根据线索得出结论
5. **感知计算** - 像艺术家一样理解图像、声音等感官信息

**认知计算的四大特点**

*适应性 - 像变色龙一样会适应*
- 能从每次经历中学到新东西
- 遇到新情况会调整自己的行为
- 就像你玩游戏时会越来越熟练
- **例子**：推荐系统会根据你的喜好变化调整推荐内容

*交互性 - 像好朋友一样会聊天*
- 能用自然语言和人类对话
- 可以同时处理文字、图片、声音等多种信息
- 就像和Siri或小爱同学聊天一样自然
- **例子**：智能客服能理解你的问题并给出合适回答

*迭代性 - 像科学家一样会实验*
- 会提出假设，然后验证是否正确
- 一步步解决复杂问题
- 根据反馈不断改进
- **例子**：医疗诊断系统会根据症状提出可能的诊断，然后根据检查结果调整

*上下文感知 - 像聪明人一样会察言观色*
- 能理解当前的环境和情况
- 根据不同场合调整行为
- 就像你在图书馆和在KTV说话声音不同
- **例子**：智能助手知道在会议中要保持安静，在家里可以正常音量回应

#### 认知架构

**CLARION架构**

*双层结构*
```python
class CLARIONArchitecture:
    def __init__(self):
        # 显式层（符号推理）
        self.explicit_layer = {
            'rules': [],
            'facts': [],
            'goals': []
        }
        
        # 隐式层（神经网络）
        self.implicit_layer = {
            'neural_networks': {},
            'associative_memory': {},
            'pattern_recognition': {}
        }
        
        # 动机子系统
        self.motivational_subsystem = {
            'drives': [],
            'goals': [],
            'emotions': {}
        }
        
        # 元认知子系统
        self.metacognitive_subsystem = {
            'monitoring': {},
            'control': {},
            'reflection': {}
        }
    
    def process_input(self, input_data, context):
        """处理输入信息"""
        # 隐式处理
        implicit_response = self.implicit_processing(input_data)
        
        # 显式推理
        explicit_response = self.explicit_reasoning(input_data, context)
        
        # 整合响应
        return self.integrate_responses(implicit_response, explicit_response)
    
    def implicit_processing(self, input_data):
        """隐式层处理"""
        # 模式识别
        patterns = self.implicit_layer['pattern_recognition'].recognize(input_data)
        
        # 联想记忆
        associations = self.implicit_layer['associative_memory'].retrieve(patterns)
        
        return {'patterns': patterns, 'associations': associations}
    
    def explicit_reasoning(self, input_data, context):
        """显式层推理"""
        # 规则匹配
        applicable_rules = self.match_rules(input_data, context)
        
        # 逻辑推理
        conclusions = self.apply_rules(applicable_rules)
        
        return {'rules': applicable_rules, 'conclusions': conclusions}
```

**ACT-R架构**

*认知模块*
```python
class ACTRArchitecture:
    def __init__(self):
        # 声明性记忆
        self.declarative_memory = {
            'chunks': {},
            'activation_levels': {},
            'retrieval_threshold': 0.0
        }
        
        # 程序性记忆
        self.procedural_memory = {
            'production_rules': [],
            'conflict_resolution': 'utility'
        }
        
        # 工作记忆缓冲区
        self.working_memory = {
            'goal_buffer': None,
            'retrieval_buffer': None,
            'visual_buffer': None,
            'manual_buffer': None
        }
    
    def cognitive_cycle(self):
        """认知循环"""
        while True:
            # 1. 模式匹配
            matching_rules = self.pattern_matching()
            
            # 2. 冲突解决
            selected_rule = self.conflict_resolution(matching_rules)
            
            # 3. 执行动作
            if selected_rule:
                self.execute_production(selected_rule)
            
            # 4. 更新激活
            self.update_activation()
    
    def pattern_matching(self):
        """模式匹配"""
        matching_rules = []
        for rule in self.procedural_memory['production_rules']:
            if self.match_conditions(rule.conditions):
                matching_rules.append(rule)
        return matching_rules
    
    def conflict_resolution(self, rules):
        """冲突解决"""
        if not rules:
            return None
        
        # 基于效用值选择
        return max(rules, key=lambda r: r.utility)
```

### 1.2 知识表示与推理

#### 语义网络

**概念图表示**
```python
class SemanticNetwork:
    def __init__(self):
        self.nodes = {}  # 概念节点
        self.edges = {}  # 关系边
    
    def add_concept(self, concept_id, properties):
        """添加概念"""
        self.nodes[concept_id] = {
            'type': 'concept',
            'properties': properties,
            'relations': []
        }
    
    def add_relation(self, source, target, relation_type, weight=1.0):
        """添加关系"""
        relation_id = f"{source}_{relation_type}_{target}"
        self.edges[relation_id] = {
            'source': source,
            'target': target,
            'type': relation_type,
            'weight': weight
        }
        
        # 更新节点关系
        if source in self.nodes:
            self.nodes[source]['relations'].append(relation_id)
    
    def spreading_activation(self, start_nodes, activation_threshold=0.1):
        """扩散激活"""
        activation = {node: 0.0 for node in self.nodes}
        
        # 初始激活
        for node in start_nodes:
            activation[node] = 1.0
        
        # 迭代扩散
        for _ in range(10):  # 最大迭代次数
            new_activation = activation.copy()
            
            for edge_id, edge in self.edges.items():
                source_activation = activation[edge['source']]
                if source_activation > activation_threshold:
                    spread_amount = source_activation * edge['weight'] * 0.1
                    new_activation[edge['target']] += spread_amount
            
            activation = new_activation
        
        return activation
    
    def find_path(self, start, end, max_depth=5):
        """寻找概念路径"""
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = set([start])
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == end:
                return path
            
            # 探索相邻节点
            for relation_id in self.nodes[current]['relations']:
                edge = self.edges[relation_id]
                next_node = edge['target']
                
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        return None
```

#### 本体工程

**OWL本体**
```python
from rdflib import Graph, Namespace, RDF, RDFS, OWL

class OntologyManager:
    def __init__(self):
        self.graph = Graph()
        self.namespace = Namespace("http://example.org/ontology#")
        
        # 绑定命名空间
        self.graph.bind("ex", self.namespace)
        self.graph.bind("owl", OWL)
    
    def create_class(self, class_name, parent_class=None):
        """创建类"""
        class_uri = self.namespace[class_name]
        self.graph.add((class_uri, RDF.type, OWL.Class))
        
        if parent_class:
            parent_uri = self.namespace[parent_class]
            self.graph.add((class_uri, RDFS.subClassOf, parent_uri))
        
        return class_uri
    
    def create_property(self, property_name, domain=None, range_=None):
        """创建属性"""
        property_uri = self.namespace[property_name]
        self.graph.add((property_uri, RDF.type, OWL.ObjectProperty))
        
        if domain:
            domain_uri = self.namespace[domain]
            self.graph.add((property_uri, RDFS.domain, domain_uri))
        
        if range_:
            range_uri = self.namespace[range_]
            self.graph.add((property_uri, RDFS.range, range_uri))
        
        return property_uri
    
    def create_individual(self, individual_name, class_name):
        """创建个体"""
        individual_uri = self.namespace[individual_name]
        class_uri = self.namespace[class_name]
        
        self.graph.add((individual_uri, RDF.type, class_uri))
        return individual_uri
    
    def sparql_query(self, query_string):
        """SPARQL查询"""
        return self.graph.query(query_string)
    
    def reasoning(self):
        """简单推理"""
        # 传递性推理
        transitive_query = """
        INSERT {
            ?x rdfs:subClassOf ?z .
        }
        WHERE {
            ?x rdfs:subClassOf ?y .
            ?y rdfs:subClassOf ?z .
            FILTER(?x != ?z)
        }
        """
        
        self.graph.update(transitive_query)
```

## 第二章：上下文建模

### 2.1 上下文理解理论

#### 上下文定义与分类

**上下文维度**
```python
class ContextModel:
    def __init__(self):
        self.dimensions = {
            'temporal': {  # 时间上下文
                'timestamp': None,
                'duration': None,
                'sequence': [],
                'periodicity': None
            },
            'spatial': {  # 空间上下文
                'location': None,
                'environment': None,
                'proximity': {},
                'topology': None
            },
            'social': {  # 社会上下文
                'participants': [],
                'relationships': {},
                'roles': {},
                'cultural_factors': {}
            },
            'task': {  # 任务上下文
                'goal': None,
                'current_task': None,
                'task_history': [],
                'constraints': []
            },
            'system': {  # 系统上下文
                'device_info': {},
                'network_status': {},
                'resource_availability': {},
                'system_state': {}
            }
        }
    
    def update_context(self, dimension, key, value):
        """更新上下文"""
        if dimension in self.dimensions:
            self.dimensions[dimension][key] = value
    
    def get_context_similarity(self, other_context):
        """计算上下文相似度"""
        similarity_scores = {}
        
        for dim in self.dimensions:
            if dim in other_context.dimensions:
                similarity_scores[dim] = self._calculate_dimension_similarity(
                    self.dimensions[dim], 
                    other_context.dimensions[dim]
                )
        
        # 加权平均
        weights = {
            'temporal': 0.2,
            'spatial': 0.2,
            'social': 0.2,
            'task': 0.3,
            'system': 0.1
        }
        
        total_similarity = sum(
            similarity_scores.get(dim, 0) * weight 
            for dim, weight in weights.items()
        )
        
        return total_similarity
    
    def _calculate_dimension_similarity(self, dim1, dim2):
        """计算维度相似度"""
        # 简化的相似度计算
        common_keys = set(dim1.keys()) & set(dim2.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0
        for key in common_keys:
            if dim1[key] == dim2[key]:
                similarity_sum += 1
        
        return similarity_sum / len(common_keys)
```

#### 上下文推理

**基于规则的上下文推理**
```python
class ContextReasoner:
    def __init__(self):
        self.rules = []
        self.facts = set()
    
    def add_rule(self, conditions, conclusions, confidence=1.0):
        """添加推理规则"""
        self.rules.append({
            'conditions': conditions,
            'conclusions': conclusions,
            'confidence': confidence
        })
    
    def add_fact(self, fact):
        """添加事实"""
        self.facts.add(fact)
    
    def forward_chaining(self):
        """前向链推理"""
        new_facts = set()
        
        for rule in self.rules:
            if self._match_conditions(rule['conditions']):
                for conclusion in rule['conclusions']:
                    if conclusion not in self.facts:
                        new_facts.add(conclusion)
                        print(f"推导出新事实: {conclusion}")
        
        if new_facts:
            self.facts.update(new_facts)
            # 递归推理
            self.forward_chaining()
    
    def _match_conditions(self, conditions):
        """匹配条件"""
        for condition in conditions:
            if condition not in self.facts:
                return False
        return True
    
    def explain_inference(self, target_fact):
        """解释推理过程"""
        explanation = []
        
        for rule in self.rules:
            if target_fact in rule['conclusions']:
                if self._match_conditions(rule['conditions']):
                    explanation.append({
                        'rule': rule,
                        'matched_conditions': rule['conditions']
                    })
        
        return explanation

# 使用示例
reasoner = ContextReasoner()

# 添加规则
reasoner.add_rule(
    conditions=['user_at_office', 'working_hours'],
    conclusions=['user_busy'],
    confidence=0.8
)

reasoner.add_rule(
    conditions=['user_busy', 'phone_ringing'],
    conclusions=['interrupt_allowed'],
    confidence=0.6
)

# 添加事实
reasoner.add_fact('user_at_office')
reasoner.add_fact('working_hours')
reasoner.add_fact('phone_ringing')

# 执行推理
reasoner.forward_chaining()
```

### 2.2 动态上下文管理

#### 上下文生命周期

**上下文管理器**
```python
import time
from collections import deque

class ContextManager:
    def __init__(self, max_history=100):
        self.current_context = ContextModel()
        self.context_history = deque(maxlen=max_history)
        self.context_stack = []  # 上下文栈
        self.listeners = []  # 上下文变化监听器
    
    def push_context(self, new_context):
        """压入新上下文"""
        # 保存当前上下文
        self.context_stack.append(self.current_context)
        self.context_history.append({
            'context': self.current_context,
            'timestamp': time.time(),
            'action': 'push'
        })
        
        # 设置新上下文
        self.current_context = new_context
        self._notify_listeners('context_pushed', new_context)
    
    def pop_context(self):
        """弹出上下文"""
        if self.context_stack:
            old_context = self.current_context
            self.current_context = self.context_stack.pop()
            
            self.context_history.append({
                'context': old_context,
                'timestamp': time.time(),
                'action': 'pop'
            })
            
            self._notify_listeners('context_popped', self.current_context)
            return old_context
        return None
    
    def update_context(self, updates):
        """更新当前上下文"""
        old_context = self.current_context
        
        for dimension, changes in updates.items():
            for key, value in changes.items():
                self.current_context.update_context(dimension, key, value)
        
        self.context_history.append({
            'context': old_context,
            'timestamp': time.time(),
            'action': 'update',
            'changes': updates
        })
        
        self._notify_listeners('context_updated', self.current_context)
    
    def add_listener(self, listener):
        """添加上下文变化监听器"""
        self.listeners.append(listener)
    
    def _notify_listeners(self, event_type, context):
        """通知监听器"""
        for listener in self.listeners:
            try:
                listener(event_type, context)
            except Exception as e:
                print(f"监听器错误: {e}")
    
    def get_context_pattern(self, time_window=3600):
        """获取上下文模式"""
        current_time = time.time()
        recent_contexts = [
            entry for entry in self.context_history
            if current_time - entry['timestamp'] <= time_window
        ]
        
        # 分析模式
        patterns = {
            'frequent_locations': {},
            'common_tasks': {},
            'typical_times': {},
            'interaction_patterns': {}
        }
        
        for entry in recent_contexts:
            context = entry['context']
            
            # 位置模式
            location = context.dimensions['spatial'].get('location')
            if location:
                patterns['frequent_locations'][location] = \
                    patterns['frequent_locations'].get(location, 0) + 1
            
            # 任务模式
            task = context.dimensions['task'].get('current_task')
            if task:
                patterns['common_tasks'][task] = \
                    patterns['common_tasks'].get(task, 0) + 1
        
        return patterns
```

## 第三章：智能推理系统

### 3.1 符号推理

#### 逻辑推理引擎

**一阶逻辑推理**
```python
class FirstOrderLogic:
    def __init__(self):
        self.predicates = {}  # 谓词定义
        self.facts = set()    # 事实库
        self.rules = []       # 规则库
    
    def add_predicate(self, name, arity):
        """添加谓词定义"""
        self.predicates[name] = arity
    
    def add_fact(self, predicate, *args):
        """添加事实"""
        if predicate in self.predicates:
            if len(args) == self.predicates[predicate]:
                fact = (predicate, args)
                self.facts.add(fact)
            else:
                raise ValueError(f"谓词 {predicate} 需要 {self.predicates[predicate]} 个参数")
    
    def add_rule(self, head, body):
        """添加规则 (head :- body)"""
        self.rules.append({
            'head': head,
            'body': body
        })
    
    def query(self, goal):
        """查询目标"""
        return self._resolve(goal, {})
    
    def _resolve(self, goal, substitution):
        """SLD解析"""
        # 检查事实
        for fact in self.facts:
            unification = self._unify(goal, fact, substitution.copy())
            if unification is not None:
                yield unification
        
        # 检查规则
        for rule in self.rules:
            # 重命名变量避免冲突
            renamed_rule = self._rename_variables(rule)
            
            unification = self._unify(goal, renamed_rule['head'], substitution.copy())
            if unification is not None:
                # 递归解析规则体
                for body_goal in renamed_rule['body']:
                    for result in self._resolve(body_goal, unification):
                        yield result
    
    def _unify(self, term1, term2, substitution):
        """合一算法"""
        # 简化的合一实现
        if term1 == term2:
            return substitution
        
        if self._is_variable(term1):
            return self._bind_variable(term1, term2, substitution)
        
        if self._is_variable(term2):
            return self._bind_variable(term2, term1, substitution)
        
        if isinstance(term1, tuple) and isinstance(term2, tuple):
            if len(term1) == len(term2):
                for t1, t2 in zip(term1, term2):
                    substitution = self._unify(t1, t2, substitution)
                    if substitution is None:
                        return None
                return substitution
        
        return None
    
    def _is_variable(self, term):
        """检查是否为变量"""
        return isinstance(term, str) and term.startswith('?')
    
    def _bind_variable(self, var, term, substitution):
        """绑定变量"""
        if var in substitution:
            return self._unify(substitution[var], term, substitution)
        else:
            substitution[var] = term
            return substitution
    
    def _rename_variables(self, rule):
        """重命名规则中的变量"""
        # 简化实现
        return rule

# 使用示例
fol = FirstOrderLogic()

# 定义谓词
fol.add_predicate('parent', 2)
fol.add_predicate('grandparent', 2)

# 添加事实
fol.add_fact('parent', 'john', 'mary')
fol.add_fact('parent', 'mary', 'susan')

# 添加规则
fol.add_rule(
    ('grandparent', ('?X', '?Z')),
    [('parent', ('?X', '?Y')), ('parent', ('?Y', '?Z'))]
)

# 查询
results = list(fol.query(('grandparent', 'john', '?who')))
print(f"查询结果: {results}")
```

### 3.2 概率推理

#### 贝叶斯网络

**贝叶斯网络实现**
```python
import numpy as np
from itertools import product

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}  # 节点信息
        self.edges = {}  # 边信息
        self.cpds = {}   # 条件概率分布
    
    def add_node(self, node_name, states):
        """添加节点"""
        self.nodes[node_name] = {
            'states': states,
            'parents': [],
            'children': []
        }
    
    def add_edge(self, parent, child):
        """添加边"""
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("节点不存在")
        
        self.nodes[parent]['children'].append(child)
        self.nodes[child]['parents'].append(parent)
        
        edge_id = f"{parent}->{child}"
        self.edges[edge_id] = {
            'parent': parent,
            'child': child
        }
    
    def set_cpd(self, node, cpd_table):
        """设置条件概率分布"""
        self.cpds[node] = cpd_table
    
    def variable_elimination(self, query_vars, evidence={}):
        """变量消除算法"""
        # 创建因子
        factors = []
        
        for node in self.nodes:
            factor = self._create_factor(node)
            factors.append(factor)
        
        # 应用证据
        for var, value in evidence.items():
            factors = [self._apply_evidence(f, var, value) for f in factors]
        
        # 消除非查询变量
        elimination_order = self._get_elimination_order(query_vars, evidence)
        
        for var in elimination_order:
            # 找到包含该变量的因子
            relevant_factors = [f for f in factors if var in f['variables']]
            other_factors = [f for f in factors if var not in f['variables']]
            
            # 合并相关因子
            if relevant_factors:
                merged_factor = self._multiply_factors(relevant_factors)
                # 边际化变量
                marginalized_factor = self._marginalize(merged_factor, var)
                factors = other_factors + [marginalized_factor]
        
        # 合并剩余因子
        result_factor = self._multiply_factors(factors)
        
        # 归一化
        return self._normalize_factor(result_factor)
    
    def _create_factor(self, node):
        """为节点创建因子"""
        parents = self.nodes[node]['parents']
        variables = parents + [node]
        
        # 创建因子表
        states_combinations = []
        for var in variables:
            states_combinations.append(self.nodes[var]['states'])
        
        factor_table = {}
        for combination in product(*states_combinations):
            # 从CPD获取概率
            prob = self._get_cpd_probability(node, combination)
            factor_table[combination] = prob
        
        return {
            'variables': variables,
            'table': factor_table
        }
    
    def _get_cpd_probability(self, node, state_combination):
        """从CPD获取概率"""
        if node in self.cpds:
            return self.cpds[node].get(state_combination, 0.0)
        return 1.0  # 默认概率
    
    def _multiply_factors(self, factors):
        """因子乘法"""
        if not factors:
            return {'variables': [], 'table': {}}
        
        if len(factors) == 1:
            return factors[0]
        
        # 合并变量
        all_variables = set()
        for factor in factors:
            all_variables.update(factor['variables'])
        all_variables = list(all_variables)
        
        # 创建新因子表
        new_table = {}
        states_combinations = []
        for var in all_variables:
            states_combinations.append(self.nodes[var]['states'])
        
        for combination in product(*states_combinations):
            prob = 1.0
            for factor in factors:
                # 提取相关状态
                relevant_states = tuple(
                    combination[all_variables.index(var)] 
                    for var in factor['variables']
                )
                prob *= factor['table'].get(relevant_states, 0.0)
            
            new_table[combination] = prob
        
        return {
            'variables': all_variables,
            'table': new_table
        }
    
    def _marginalize(self, factor, variable):
        """边际化变量"""
        if variable not in factor['variables']:
            return factor
        
        var_index = factor['variables'].index(variable)
        new_variables = [v for v in factor['variables'] if v != variable]
        
        new_table = {}
        for state_combination, prob in factor['table'].items():
            # 移除被边际化的变量
            new_combination = tuple(
                state_combination[i] for i in range(len(state_combination))
                if i != var_index
            )
            
            if new_combination in new_table:
                new_table[new_combination] += prob
            else:
                new_table[new_combination] = prob
        
        return {
            'variables': new_variables,
            'table': new_table
        }
    
    def _get_elimination_order(self, query_vars, evidence):
        """获取消除顺序"""
        all_vars = set(self.nodes.keys())
        eliminate_vars = all_vars - set(query_vars) - set(evidence.keys())
        return list(eliminate_vars)
    
    def _apply_evidence(self, factor, evidence_var, evidence_value):
        """应用证据"""
        if evidence_var not in factor['variables']:
            return factor
        
        var_index = factor['variables'].index(evidence_var)
        new_table = {}
        
        for state_combination, prob in factor['table'].items():
            if state_combination[var_index] == evidence_value:
                new_table[state_combination] = prob
        
        return {
            'variables': factor['variables'],
            'table': new_table
        }
    
    def _normalize_factor(self, factor):
        """归一化因子"""
        total = sum(factor['table'].values())
        if total > 0:
            normalized_table = {
                k: v / total for k, v in factor['table'].items()
            }
            return {
                'variables': factor['variables'],
                'table': normalized_table
            }
        return factor
```

## 实践项目

### 项目一：智能问答系统

**系统架构**
```python
class IntelligentQASystem:
    def __init__(self):
        self.knowledge_base = SemanticNetwork()
        self.context_manager = ContextManager()
        self.reasoner = ContextReasoner()
        self.nlp_processor = None  # NLP处理器
    
    def process_question(self, question, user_context=None):
        """处理问题"""
        # 1. 更新上下文
        if user_context:
            self.context_manager.update_context(user_context)
        
        # 2. 问题理解
        parsed_question = self.parse_question(question)
        
        # 3. 知识检索
        relevant_knowledge = self.retrieve_knowledge(parsed_question)
        
        # 4. 推理生成答案
        answer = self.generate_answer(parsed_question, relevant_knowledge)
        
        # 5. 上下文更新
        self.update_conversation_context(question, answer)
        
        return answer
    
    def parse_question(self, question):
        """解析问题"""
        # 简化的问题解析
        return {
            'text': question,
            'type': self.classify_question_type(question),
            'entities': self.extract_entities(question),
            'intent': self.extract_intent(question)
        }
    
    def retrieve_knowledge(self, parsed_question):
        """检索相关知识"""
        entities = parsed_question['entities']
        
        # 使用扩散激活检索
        activation = self.knowledge_base.spreading_activation(entities)
        
        # 选择高激活度的概念
        relevant_concepts = [
            concept for concept, score in activation.items()
            if score > 0.1
        ]
        
        return relevant_concepts
    
    def generate_answer(self, question, knowledge):
        """生成答案"""
        # 基于模板的答案生成
        question_type = question['type']
        
        if question_type == 'factual':
            return self.generate_factual_answer(question, knowledge)
        elif question_type == 'procedural':
            return self.generate_procedural_answer(question, knowledge)
        else:
            return "我需要更多信息来回答这个问题。"
    
    def generate_factual_answer(self, question, knowledge):
        """生成事实性答案"""
        # 查找相关事实
        facts = []
        for concept in knowledge:
            if concept in self.knowledge_base.nodes:
                facts.extend(
                    self.knowledge_base.nodes[concept]['properties']
                )
        
        if facts:
            return f"根据我的知识，{facts[0]}"
        else:
            return "抱歉，我没有找到相关信息。"
```

### 项目二：上下文感知推荐系统

**推荐引擎**
```python
class ContextAwareRecommender:
    def __init__(self):
        self.user_profiles = {}  # 用户画像
        self.item_features = {}  # 物品特征
        self.context_model = ContextModel()
        self.interaction_history = []
    
    def recommend(self, user_id, context, num_recommendations=10):
        """生成推荐"""
        # 1. 更新用户上下文
        self.context_model = context
        
        # 2. 获取候选物品
        candidates = self.get_candidate_items(user_id)
        
        # 3. 上下文感知评分
        scored_items = []
        for item in candidates:
            score = self.calculate_contextual_score(user_id, item, context)
            scored_items.append((item, score))
        
        # 4. 排序并返回推荐
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in scored_items[:num_recommendations]]
    
    def calculate_contextual_score(self, user_id, item_id, context):
        """计算上下文感知评分"""
        # 基础评分
        base_score = self.calculate_base_score(user_id, item_id)
        
        # 上下文调整因子
        context_factors = {
            'temporal': self.get_temporal_factor(item_id, context),
            'spatial': self.get_spatial_factor(item_id, context),
            'social': self.get_social_factor(item_id, context),
            'mood': self.get_mood_factor(item_id, context)
        }
        
        # 综合评分
        contextual_score = base_score
        for factor_name, factor_value in context_factors.items():
            contextual_score *= factor_value
        
        return contextual_score
    
    def get_temporal_factor(self, item_id, context):
        """时间上下文因子"""
        current_time = context.dimensions['temporal']['timestamp']
        
        # 根据时间调整推荐权重
        if 'time_preferences' in self.item_features.get(item_id, {}):
            preferred_times = self.item_features[item_id]['time_preferences']
            
            import datetime
            current_hour = datetime.datetime.fromtimestamp(current_time).hour
            
            if current_hour in preferred_times:
                return 1.2  # 提高权重
            else:
                return 0.8  # 降低权重
        
        return 1.0  # 默认权重
    
    def get_spatial_factor(self, item_id, context):
        """空间上下文因子"""
        user_location = context.dimensions['spatial']['location']
        
        if 'location_relevance' in self.item_features.get(item_id, {}):
            item_locations = self.item_features[item_id]['location_relevance']
            
            if user_location in item_locations:
                return item_locations[user_location]
        
        return 1.0
    
    def update_interaction(self, user_id, item_id, interaction_type, context):
        """更新交互历史"""
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'context': context,
            'timestamp': time.time()
        }
        
        self.interaction_history.append(interaction)
        
        # 更新用户画像
        self.update_user_profile(user_id, interaction)
    
    def update_user_profile(self, user_id, interaction):
        """更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'preferences': {},
                'context_patterns': {},
                'interaction_count': 0
            }
        
        profile = self.user_profiles[user_id]
        profile['interaction_count'] += 1
        
        # 更新偏好
        item_features = self.item_features.get(interaction['item_id'], {})
        for feature, value in item_features.items():
            if feature not in profile['preferences']:
                profile['preferences'][feature] = {}
            
            if value not in profile['preferences'][feature]:
                profile['preferences'][feature][value] = 0
            
            profile['preferences'][feature][value] += 1
```

### 项目三：认知助手系统

**认知助手架构**
```python
class CognitiveAssistant:
    def __init__(self):
        self.cognitive_architecture = CLARIONArchitecture()
        self.context_manager = ContextManager()
        self.knowledge_base = OntologyManager()
        self.learning_module = None
        self.dialogue_manager = None
    
    def process_user_input(self, user_input, context):
        """处理用户输入"""
        # 1. 上下文更新
        self.context_manager.update_context(context)
        
        # 2. 输入理解
        understood_input = self.understand_input(user_input)
        
        # 3. 认知处理
        cognitive_response = self.cognitive_architecture.process_input(
            understood_input, 
            self.context_manager.current_context
        )
        
        # 4. 响应生成
        response = self.generate_response(cognitive_response)
        
        # 5. 学习更新
        self.update_learning(user_input, response, context)
        
        return response
    
    def understand_input(self, user_input):
        """理解用户输入"""
        return {
            'text': user_input,
            'intent': self.extract_intent(user_input),
            'entities': self.extract_entities(user_input),
            'sentiment': self.analyze_sentiment(user_input),
            'complexity': self.assess_complexity(user_input)
        }
    
    def generate_response(self, cognitive_response):
        """生成响应"""
        # 基于认知处理结果生成自然语言响应
        if 'conclusions' in cognitive_response:
            conclusions = cognitive_response['conclusions']
            return self.verbalize_conclusions(conclusions)
        else:
            return "我需要更多信息来帮助您。"
    
    def update_learning(self, input_data, response, context):
        """更新学习"""
        # 记录交互经验
        experience = {
            'input': input_data,
            'context': context,
            'response': response,
            'timestamp': time.time()
        }
        
        # 更新认知架构
        self.cognitive_architecture.learn_from_experience(experience)
    
    def proactive_assistance(self):
        """主动协助"""
        # 分析用户行为模式
        patterns = self.context_manager.get_context_pattern()
        
        # 预测用户需求
        predicted_needs = self.predict_user_needs(patterns)
        
        # 生成主动建议
        suggestions = []
        for need in predicted_needs:
            suggestion = self.generate_suggestion(need)
            suggestions.append(suggestion)
        
        return suggestions
    
    def predict_user_needs(self, patterns):
        """预测用户需求"""
        needs = []
        
        # 基于时间模式预测
        current_time = time.time()
        hour = time.localtime(current_time).tm_hour
        
        if hour == 9 and 'office' in patterns.get('frequent_locations', {}):
            needs.append('daily_schedule_review')
        
        if hour == 17 and 'commute' in patterns.get('common_tasks', {}):
            needs.append('traffic_update')
        
        return needs
```

## 学习评估

### 理论评估
1. **认知计算原理**（25分）
   - 认知架构理解
   - 知识表示方法
   - 推理机制原理

2. **上下文建模**（25分）
   - 上下文维度分析
   - 上下文推理方法
   - 动态上下文管理

3. **智能推理**（25分）
   - 符号推理实现
   - 概率推理应用
   - 混合推理策略

### 实践评估
1. **智能问答系统**（25分）
   - 系统架构设计
   - 知识库构建
   - 推理引擎实现
   - 上下文处理能力

2. **推荐系统**（25分）
   - 上下文感知机制
   - 推荐算法实现
   - 用户画像建模
   - 系统性能评估

3. **认知助手**（25分）
   - 认知架构实现
   - 多模态交互
   - 学习适应能力
   - 主动服务功能

### 综合评估
1. **创新性**（25分）
   - 技术创新点
   - 应用场景创新
   - 解决方案独特性

2. **实用性**（25分）
   - 实际应用价值
   - 用户体验设计
   - 系统可扩展性

## 延伸学习

### 前沿研究方向
1. **神经符号计算**
   - 神经网络与符号推理结合
   - 可解释AI系统
   - 知识图谱嵌入

2. **因果推理**
   - 因果发现算法
   - 反事实推理
   - 因果表示学习

3. **元认知学习**
   - 学会学习
   - 认知策略优化
   - 自适应推理

### 应用领域
1. **智能教育**
   - 个性化学习系统
   - 认知诊断
   - 智能辅导

2. **智能医疗**
   - 临床决策支持
   - 医学知识推理
   - 个性化治疗

3. **智能制造**
   - 工业知识图谱
   - 故障诊断推理
   - 智能运维

### 工具和框架
1. **知识图谱工具**
   - Neo4j
   - Apache Jena
   - RDFLib

2. **推理引擎**
   - Prolog
   - CLIPS
   - Drools

3. **认知架构**
   - ACT-R
   - SOAR
   - CLARION

## 总结

本模块深入探讨了认知计算与上下文理解的核心技术，包括：

1. **认知计算基础**：理解了认知计算的基本概念、架构和原理
2. **知识表示与推理**：掌握了语义网络、本体工程等知识表示方法
3. **上下文建模**：学习了上下文的多维度建模和动态管理
4. **智能推理系统**：实现了符号推理和概率推理的结合
5. **实践应用**：通过三个综合项目加深了对技术的理解和应用

这些技术为构建更加智能、自适应的AI系统提供了重要基础，是实现真正智能化应用的关键技术。