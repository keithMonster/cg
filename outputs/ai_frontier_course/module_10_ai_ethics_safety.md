# 模块十：AI伦理与安全

## 学习目标

通过本模块的学习，您将能够：

1. **理解AI伦理的核心概念**：掌握AI伦理的基本原则、道德框架和价值体系
2. **识别AI系统中的偏见**：学会检测、分析和缓解算法偏见与歧视问题
3. **评估AI安全风险**：了解AI系统的安全威胁、攻击方式和防护策略
4. **设计负责任的AI**：掌握可解释AI、公平性设计和隐私保护技术
5. **应用伦理决策框架**：能够在AI开发和部署中进行伦理评估和决策

## 第一节：AI伦理基础理论

### 1.1 伦理学基础

#### 道德哲学框架

**什么是AI伦理？**

AI伦理就像是给AI制定的"道德准则"，确保AI在帮助人类的同时，不会伤害到任何人。就像我们教育孩子要诚实、善良一样，我们也需要教AI做"正确的事"。

**两大伦理思想流派**

**功利主义 - 像精明的管家**
- **核心思想**：做能让大多数人受益最多的事
- **判断标准**：看结果，不看过程
- **生活例子**：
  - 一个药物能救100个人，但会让1个人不舒服，功利主义会选择使用这个药物
  - 自动驾驶汽车在紧急情况下选择撞向人少的一边
- **在AI中的应用**：
  - 推荐系统优化整体用户满意度
  - 资源分配算法让更多人受益
  - 医疗AI优先救治更多病人

**举个例子**：
假设一个AI客服系统有两种设计方案：
- 方案A：让90%的用户满意，但10%的用户很不满意
- 方案B：让70%的用户比较满意，30%的用户稍微不满意
功利主义会选择方案A，因为总体满意度更高

```python
class UtilitarianEthics:
    """功利主义伦理框架"""
    
    def __init__(self):
        self.stakeholders = []
        self.outcomes = {}
    
    def add_stakeholder(self, name, weight=1.0):
        """添加利益相关者"""
        self.stakeholders.append({
            'name': name,
            'weight': weight
        })
    
    def evaluate_outcome(self, action, stakeholder_impacts):
        """评估行动的功利主义价值"""
        total_utility = 0
        
        for stakeholder in self.stakeholders:
            name = stakeholder['name']
            weight = stakeholder['weight']
            
            if name in stakeholder_impacts:
                impact = stakeholder_impacts[name]
                weighted_impact = impact * weight
                total_utility += weighted_impact
        
        return total_utility
    
    def compare_actions(self, actions_impacts):
        """比较多个行动的伦理价值"""
        results = {}
        
        for action, impacts in actions_impacts.items():
            utility = self.evaluate_outcome(action, impacts)
            results[action] = utility
        
        # 返回效用最高的行动
        best_action = max(results, key=results.get)
        return best_action, results

# 使用示例
ethics = UtilitarianEthics()

# 添加利益相关者
ethics.add_stakeholder('用户', weight=1.0)
ethics.add_stakeholder('企业', weight=0.8)
ethics.add_stakeholder('社会', weight=1.2)

# 评估不同AI决策的影响
actions = {
    '个性化推荐': {
        '用户': 0.8,    # 用户体验提升
        '企业': 0.9,    # 收入增加
        '社会': -0.3    # 可能加剧信息茧房
    },
    '隐私保护': {
        '用户': 0.7,    # 隐私安全
        '企业': -0.4,   # 收入减少
        '社会': 0.6     # 社会信任提升
    }
}

best_action, utilities = ethics.compare_actions(actions)
print(f"最佳行动: {best_action}")
print(f"各行动效用: {utilities}")
```

**义务伦理学 - 像严格的法官**
- **核心思想**：有些事情无论如何都不能做，有些规则必须遵守
- **判断标准**：看行为本身是否正确，不管结果如何
- **生活例子**：
  - "不能撒谎"是绝对规则，即使撒谎能救人也不行
  - "不能偷盗"是绝对规则，即使是为了帮助穷人也不行
- **在AI中的应用**：
  - AI绝对不能侵犯用户隐私，即使这样能提供更好服务
  - AI不能歧视任何群体，即使某些歧视在统计上"有道理"
  - AI不能欺骗用户，即使善意的谎言能让用户更开心

**举个例子**：
同样是AI客服系统的例子：
- 如果系统需要收集用户隐私数据才能提供更好服务
- 功利主义可能会说："收集吧，大家都受益"
- 义务伦理学会说："不行，侵犯隐私本身就是错的"

**两种伦理观的对比**：
- **功利主义**：像商人，追求最大利益
- **义务伦理学**：像原则性很强的人，坚持底线

```python
class DeontologicalEthics:
    """义务伦理学框架"""
    
    def __init__(self):
        self.moral_rules = []
        self.categorical_imperatives = []
    
    def add_moral_rule(self, rule, priority=1):
        """添加道德规则"""
        self.moral_rules.append({
            'rule': rule,
            'priority': priority
        })
    
    def add_categorical_imperative(self, imperative):
        """添加绝对命令"""
        self.categorical_imperatives.append(imperative)
    
    def evaluate_action(self, action_description):
        """评估行动的道德性"""
        violations = []
        
        # 检查是否违反道德规则
        for rule in self.moral_rules:
            if self._violates_rule(action_description, rule['rule']):
                violations.append({
                    'type': 'moral_rule',
                    'rule': rule['rule'],
                    'priority': rule['priority']
                })
        
        # 检查是否违反绝对命令
        for imperative in self.categorical_imperatives:
            if self._violates_imperative(action_description, imperative):
                violations.append({
                    'type': 'categorical_imperative',
                    'imperative': imperative,
                    'priority': 10  # 最高优先级
                })
        
        return {
            'is_moral': len(violations) == 0,
            'violations': violations,
            'severity': sum(v['priority'] for v in violations)
        }
    
    def _violates_rule(self, action, rule):
        """检查行动是否违反规则（简化实现）"""
        # 这里应该有更复杂的逻辑来判断
        keywords = rule.lower().split()
        action_lower = action.lower()
        
        # 简单的关键词匹配
        if '不得' in rule or '禁止' in rule:
            for keyword in keywords:
                if keyword in action_lower and keyword not in ['不得', '禁止']:
                    return True
        
        return False
    
    def _violates_imperative(self, action, imperative):
        """检查是否违反绝对命令"""
        # 康德的普遍化原则：如果所有人都这样做会怎样？
        return self._violates_rule(action, imperative)

# 使用示例
deontological = DeontologicalEthics()

# 添加道德规则
deontological.add_moral_rule('不得欺骗用户', priority=8)
deontological.add_moral_rule('不得侵犯隐私', priority=9)
deontological.add_moral_rule('不得歧视任何群体', priority=10)

# 添加绝对命令
deontological.add_categorical_imperative('将人当作目的而非手段')

# 评估AI行动
actions_to_evaluate = [
    '收集用户数据用于广告投放',
    '基于种族进行风险评估',
    '提供透明的算法解释'
]

for action in actions_to_evaluate:
    result = deontological.evaluate_action(action)
    print(f"行动: {action}")
    print(f"道德性: {'符合' if result['is_moral'] else '违反'}")
    if result['violations']:
        print(f"违反项: {[v['rule'] if 'rule' in v else v['imperative'] for v in result['violations']]}")
    print(f"严重程度: {result['severity']}\n")
```

**美德伦理学（Virtue Ethics）**
- 关注品格和美德
- 强调道德行为者的品质
- 在AI中的应用：培养负责任的AI开发文化

```python
class VirtueEthics:
    """美德伦理学框架"""
    
    def __init__(self):
        self.virtues = {
            '诚实': {'weight': 1.0, 'description': '提供真实、准确的信息'},
            '公正': {'weight': 1.0, 'description': '公平对待所有用户'},
            '仁慈': {'weight': 0.9, 'description': '以用户福祉为重'},
            '谦逊': {'weight': 0.8, 'description': '承认AI系统的局限性'},
            '责任': {'weight': 1.0, 'description': '对AI系统的影响负责'},
            '透明': {'weight': 0.9, 'description': '提供可理解的解释'}
        }
    
    def evaluate_virtue_alignment(self, action_description, virtue_scores):
        """评估行动与美德的一致性"""
        total_score = 0
        total_weight = 0
        
        virtue_analysis = {}
        
        for virtue, config in self.virtues.items():
            if virtue in virtue_scores:
                score = virtue_scores[virtue]  # 0-1之间的分数
                weight = config['weight']
                
                weighted_score = score * weight
                total_score += weighted_score
                total_weight += weight
                
                virtue_analysis[virtue] = {
                    'score': score,
                    'weight': weight,
                    'weighted_score': weighted_score,
                    'description': config['description']
                }
        
        overall_virtue_score = total_score / total_weight if total_weight > 0 else 0
        
        return {
            'overall_score': overall_virtue_score,
            'virtue_breakdown': virtue_analysis,
            'recommendation': self._get_virtue_recommendation(overall_virtue_score)
        }
    
    def _get_virtue_recommendation(self, score):
        """根据美德分数提供建议"""
        if score >= 0.8:
            return '行动体现了高度的美德，值得推荐'
        elif score >= 0.6:
            return '行动基本符合美德要求，可以改进'
        elif score >= 0.4:
            return '行动在美德方面有所欠缺，需要重新考虑'
        else:
            return '行动严重违背美德原则，不应执行'

# 使用示例
virtue_ethics = VirtueEthics()

# 评估AI推荐系统的美德表现
recommendation_system_virtues = {
    '诚实': 0.7,    # 推荐基于真实偏好，但可能有商业考量
    '公正': 0.6,    # 对大部分用户公平，但可能对小众群体不够友好
    '仁慈': 0.8,    # 主要考虑用户体验
    '谦逊': 0.5,    # 很少承认推荐的不确定性
    '责任': 0.7,    # 有一定的责任机制
    '透明': 0.4     # 算法黑盒，解释性不足
}

result = virtue_ethics.evaluate_virtue_alignment(
    '个性化推荐系统', 
    recommendation_system_virtues
)

print(f"整体美德分数: {result['overall_score']:.2f}")
print(f"建议: {result['recommendation']}")
print("\n各美德详细分析:")
for virtue, analysis in result['virtue_breakdown'].items():
    print(f"{virtue}: {analysis['score']:.1f} (权重: {analysis['weight']}) - {analysis['description']}")
```

### 1.2 AI伦理原则

#### 核心伦理原则

```python
class AIEthicsPrinciples:
    """AI伦理原则框架"""
    
    def __init__(self):
        self.principles = {
            'beneficence': {
                'name': '有益性',
                'description': 'AI应该促进人类福祉',
                'criteria': [
                    '提升生活质量',
                    '解决实际问题',
                    '创造积极价值'
                ]
            },
            'non_maleficence': {
                'name': '无害性',
                'description': 'AI不应造成伤害',
                'criteria': [
                    '避免身体伤害',
                    '防止心理伤害',
                    '减少社会负面影响'
                ]
            },
            'autonomy': {
                'name': '自主性',
                'description': '尊重人类的自主决策权',
                'criteria': [
                    '保持人类控制',
                    '提供选择权',
                    '支持知情决策'
                ]
            },
            'justice': {
                'name': '公正性',
                'description': '公平分配AI的利益和风险',
                'criteria': [
                    '平等获取机会',
                    '公平对待所有群体',
                    '合理分配资源'
                ]
            },
            'explicability': {
                'name': '可解释性',
                'description': 'AI决策应该可以理解和解释',
                'criteria': [
                    '提供决策理由',
                    '使用可理解的语言',
                    '支持质疑和申诉'
                ]
            }
        }
    
    def evaluate_ai_system(self, system_description, principle_scores):
        """评估AI系统的伦理合规性"""
        evaluation_results = {}
        total_score = 0
        
        for principle_id, principle_info in self.principles.items():
            if principle_id in principle_scores:
                score = principle_scores[principle_id]
                
                evaluation_results[principle_id] = {
                    'name': principle_info['name'],
                    'score': score,
                    'description': principle_info['description'],
                    'criteria': principle_info['criteria'],
                    'status': self._get_compliance_status(score)
                }
                
                total_score += score
        
        average_score = total_score / len(principle_scores) if principle_scores else 0
        
        return {
            'system': system_description,
            'overall_score': average_score,
            'overall_status': self._get_compliance_status(average_score),
            'principle_evaluations': evaluation_results,
            'recommendations': self._generate_recommendations(evaluation_results)
        }
    
    def _get_compliance_status(self, score):
        """根据分数确定合规状态"""
        if score >= 0.8:
            return '优秀'
        elif score >= 0.6:
            return '良好'
        elif score >= 0.4:
            return '需要改进'
        else:
            return '不合规'
    
    def _generate_recommendations(self, evaluations):
        """生成改进建议"""
        recommendations = []
        
        for principle_id, eval_result in evaluations.items():
            if eval_result['score'] < 0.6:
                recommendations.append({
                    'principle': eval_result['name'],
                    'current_score': eval_result['score'],
                    'suggestion': f"需要加强{eval_result['name']}方面的设计和实现",
                    'criteria': eval_result['criteria']
                })
        
        return recommendations

# 使用示例
ai_ethics = AIEthicsPrinciples()

# 评估一个医疗诊断AI系统
medical_ai_scores = {
    'beneficence': 0.9,      # 高度有益，帮助诊断疾病
    'non_maleficence': 0.7,  # 大部分情况安全，但可能误诊
    'autonomy': 0.6,         # 医生仍有最终决策权，但影响较大
    'justice': 0.5,          # 可能对某些群体有偏见
    'explicability': 0.4     # 深度学习模型，解释性不足
}

evaluation = ai_ethics.evaluate_ai_system(
    '医疗诊断AI系统', 
    medical_ai_scores
)

print(f"系统: {evaluation['system']}")
print(f"总体评分: {evaluation['overall_score']:.2f} ({evaluation['overall_status']})")
print("\n各原则评估:")
for principle_id, result in evaluation['principle_evaluations'].items():
    print(f"{result['name']}: {result['score']:.1f} - {result['status']}")

print("\n改进建议:")
for rec in evaluation['recommendations']:
    print(f"- {rec['principle']}: {rec['suggestion']}")
```

## 第二节：算法偏见与公平性

### 2.1 偏见的类型与来源

#### 数据偏见检测

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BiasDetector:
    """算法偏见检测器"""
    
    def __init__(self):
        self.bias_metrics = {}
        self.protected_attributes = []
    
    def add_protected_attribute(self, attribute_name):
        """添加受保护属性"""
        self.protected_attributes.append(attribute_name)
    
    def detect_statistical_parity(self, y_pred, protected_attr):
        """检测统计均等性"""
        groups = np.unique(protected_attr)
        positive_rates = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_predictions = y_pred[group_mask]
            positive_rate = np.mean(group_predictions)
            positive_rates[group] = positive_rate
        
        # 计算最大差异
        rates = list(positive_rates.values())
        max_diff = max(rates) - min(rates)
        
        return {
            'metric': 'statistical_parity',
            'group_rates': positive_rates,
            'max_difference': max_diff,
            'is_fair': max_diff < 0.1  # 10%阈值
        }
    
    def detect_equalized_odds(self, y_true, y_pred, protected_attr):
        """检测机会均等性"""
        groups = np.unique(protected_attr)
        tpr_by_group = {}  # True Positive Rate
        fpr_by_group = {}  # False Positive Rate
        
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # 计算TPR和FPR
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_by_group[group] = tpr
            fpr_by_group[group] = fpr
        
        # 计算TPR和FPR的最大差异
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        tpr_diff = max(tpr_values) - min(tpr_values)
        fpr_diff = max(fpr_values) - min(fpr_values)
        
        return {
            'metric': 'equalized_odds',
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'is_fair': tpr_diff < 0.1 and fpr_diff < 0.1
        }
    
    def detect_calibration(self, y_true, y_prob, protected_attr, n_bins=10):
        """检测校准公平性"""
        groups = np.unique(protected_attr)
        calibration_by_group = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            # 将概率分箱
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            bin_info = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = group_y_true[in_bin].mean()
                    avg_confidence_in_bin = group_y_prob[in_bin].mean()
                    
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    bin_info.append({
                        'bin_range': (bin_lower, bin_upper),
                        'accuracy': accuracy_in_bin,
                        'confidence': avg_confidence_in_bin,
                        'proportion': prop_in_bin
                    })
            
            calibration_by_group[group] = {
                'calibration_error': calibration_error,
                'bin_info': bin_info
            }
        
        # 计算组间校准差异
        errors = [info['calibration_error'] for info in calibration_by_group.values()]
        max_error_diff = max(errors) - min(errors)
        
        return {
            'metric': 'calibration',
            'calibration_by_group': calibration_by_group,
            'max_error_difference': max_error_diff,
            'is_fair': max_error_diff < 0.05
        }
    
    def comprehensive_bias_audit(self, X, y, protected_attrs, model=None):
        """综合偏见审计"""
        if model is None:
            model = RandomForestClassifier(random_state=42)
        
        # 训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        audit_results = {
            'model_accuracy': accuracy_score(y_test, y_pred),
            'bias_analysis': {}
        }
        
        # 对每个受保护属性进行偏见分析
        for attr_name in protected_attrs:
            if attr_name in X.columns:
                attr_values = X_test[attr_name].values
                
                # 统计均等性
                stat_parity = self.detect_statistical_parity(y_pred, attr_values)
                
                # 机会均等性
                eq_odds = self.detect_equalized_odds(y_test, y_pred, attr_values)
                
                # 校准公平性
                calibration = self.detect_calibration(y_test, y_prob, attr_values)
                
                audit_results['bias_analysis'][attr_name] = {
                    'statistical_parity': stat_parity,
                    'equalized_odds': eq_odds,
                    'calibration': calibration
                }
        
        return audit_results
    
    def visualize_bias_results(self, audit_results):
        """可视化偏见分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI系统偏见分析报告', fontsize=16)
        
        # 1. 统计均等性可视化
        ax1 = axes[0, 0]
        for attr_name, analysis in audit_results['bias_analysis'].items():
            stat_parity = analysis['statistical_parity']
            groups = list(stat_parity['group_rates'].keys())
            rates = list(stat_parity['group_rates'].values())
            
            ax1.bar([f"{attr_name}_{g}" for g in groups], rates, alpha=0.7)
        
        ax1.set_title('统计均等性 - 各组正例率')
        ax1.set_ylabel('正例率')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 机会均等性可视化
        ax2 = axes[0, 1]
        tpr_data = []
        fpr_data = []
        labels = []
        
        for attr_name, analysis in audit_results['bias_analysis'].items():
            eq_odds = analysis['equalized_odds']
            for group, tpr in eq_odds['tpr_by_group'].items():
                tpr_data.append(tpr)
                fpr_data.append(eq_odds['fpr_by_group'][group])
                labels.append(f"{attr_name}_{group}")
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax2.bar(x - width/2, tpr_data, width, label='TPR', alpha=0.7)
        ax2.bar(x + width/2, fpr_data, width, label='FPR', alpha=0.7)
        ax2.set_title('机会均等性 - TPR vs FPR')
        ax2.set_ylabel('率')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.legend()
        
        # 3. 校准误差可视化
        ax3 = axes[1, 0]
        calibration_errors = []
        group_labels = []
        
        for attr_name, analysis in audit_results['bias_analysis'].items():
            calibration = analysis['calibration']
            for group, cal_info in calibration['calibration_by_group'].items():
                calibration_errors.append(cal_info['calibration_error'])
                group_labels.append(f"{attr_name}_{group}")
        
        ax3.bar(group_labels, calibration_errors, alpha=0.7)
        ax3.set_title('校准公平性 - 校准误差')
        ax3.set_ylabel('校准误差')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 综合公平性评分
        ax4 = axes[1, 1]
        fairness_scores = []
        metrics = ['统计均等性', '机会均等性', '校准公平性']
        
        for attr_name, analysis in audit_results['bias_analysis'].items():
            stat_fair = 1 if analysis['statistical_parity']['is_fair'] else 0
            eq_fair = 1 if analysis['equalized_odds']['is_fair'] else 0
            cal_fair = 1 if analysis['calibration']['is_fair'] else 0
            
            fairness_scores.append([stat_fair, eq_fair, cal_fair])
        
        if fairness_scores:
            fairness_matrix = np.array(fairness_scores).T
            sns.heatmap(fairness_matrix, 
                       xticklabels=list(audit_results['bias_analysis'].keys()),
                       yticklabels=metrics,
                       annot=True, 
                       cmap='RdYlGn',
                       ax=ax4)
            ax4.set_title('公平性合规矩阵')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# 使用示例：生成模拟数据
np.random.seed(42)
n_samples = 1000

# 创建模拟数据集（招聘场景）
data = {
    'education_score': np.random.normal(75, 15, n_samples),
    'experience_years': np.random.exponential(5, n_samples),
    'interview_score': np.random.normal(70, 20, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
    'age_group': np.random.choice(['young', 'middle', 'senior'], n_samples, p=[0.4, 0.4, 0.2])
}

# 引入偏见：女性和年长者获得工作的概率较低
hire_prob = 0.3 + 0.01 * data['education_score'] + 0.02 * data['experience_years'] + 0.005 * data['interview_score']
hire_prob = np.where(data['gender'] == 'F', hire_prob * 0.8, hire_prob)  # 性别偏见
hire_prob = np.where(data['age_group'] == 'senior', hire_prob * 0.7, hire_prob)  # 年龄偏见
hire_prob = np.clip(hire_prob, 0, 1)

data['hired'] = np.random.binomial(1, hire_prob, n_samples)

df = pd.DataFrame(data)

# 进行偏见检测
bias_detector = BiasDetector()
bias_detector.add_protected_attribute('gender')
bias_detector.add_protected_attribute('age_group')

# 准备特征和标签
X = pd.get_dummies(df[['education_score', 'experience_years', 'interview_score', 'gender', 'age_group']])
y = df['hired']

# 执行综合偏见审计
audit_results = bias_detector.comprehensive_bias_audit(
    X, y, 
    protected_attrs=['gender_F', 'age_group_senior']
)

# 打印结果
print(f"模型准确率: {audit_results['model_accuracy']:.3f}")
print("\n偏见分析结果:")

for attr_name, analysis in audit_results['bias_analysis'].items():
    print(f"\n{attr_name}:")
    
    # 统计均等性
    stat_parity = analysis['statistical_parity']
    print(f"  统计均等性: {'通过' if stat_parity['is_fair'] else '未通过'}")
    print(f"    最大差异: {stat_parity['max_difference']:.3f}")
    
    # 机会均等性
    eq_odds = analysis['equalized_odds']
    print(f"  机会均等性: {'通过' if eq_odds['is_fair'] else '未通过'}")
    print(f"    TPR差异: {eq_odds['tpr_difference']:.3f}")
    print(f"    FPR差异: {eq_odds['fpr_difference']:.3f}")
    
    # 校准公平性
    calibration = analysis['calibration']
    print(f"  校准公平性: {'通过' if calibration['is_fair'] else '未通过'}")
    print(f"    最大校准误差差异: {calibration['max_error_difference']:.3f}")

# 可视化结果
bias_detector.visualize_bias_results(audit_results)
```

### 2.2 偏见缓解技术

#### 预处理方法

```python
class BiasPreprocessor:
    """偏见预处理器"""
    
    def __init__(self):
        self.reweighting_weights = {}
        self.synthetic_samples = None
    
    def reweighting(self, X, y, protected_attr):
        """重新加权方法"""
        # 计算各组合的权重
        unique_combinations = []
        for prot_val in np.unique(protected_attr):
            for y_val in np.unique(y):
                unique_combinations.append((prot_val, y_val))
        
        # 计算期望概率（假设各组应该有相同的正例率）
        overall_positive_rate = np.mean(y)
        
        weights = np.ones(len(X))
        
        for prot_val, y_val in unique_combinations:
            mask = (protected_attr == prot_val) & (y == y_val)
            count = np.sum(mask)
            
            if count > 0:
                if y_val == 1:  # 正例
                    expected_count = np.sum(protected_attr == prot_val) * overall_positive_rate
                else:  # 负例
                    expected_count = np.sum(protected_attr == prot_val) * (1 - overall_positive_rate)
                
                weight = expected_count / count if count > 0 else 1
                weights[mask] = weight
        
        self.reweighting_weights = weights
        return weights
    
    def disparate_impact_remover(self, X, protected_attr, repair_level=1.0):
        """差异影响消除器"""
        X_repaired = X.copy()
        
        # 对每个非受保护特征进行修复
        for col in X.columns:
            if col not in protected_attr:
                # 计算各受保护组的特征分布
                group_means = {}
                for group in np.unique(X[protected_attr]):
                    group_mask = X[protected_attr] == group
                    group_means[group] = X.loc[group_mask, col].mean()
                
                # 计算全局均值
                global_mean = X[col].mean()
                
                # 修复特征值
                for group in group_means:
                    group_mask = X[protected_attr] == group
                    original_values = X.loc[group_mask, col]
                    
                    # 线性插值修复
                    repaired_values = (
                        (1 - repair_level) * original_values + 
                        repair_level * global_mean
                    )
                    
                    X_repaired.loc[group_mask, col] = repaired_values
        
        return X_repaired
    
    def synthetic_minority_oversampling(self, X, y, protected_attr, k_neighbors=5):
        """合成少数群体过采样"""
        from sklearn.neighbors import NearestNeighbors
        
        # 识别需要过采样的群体
        minority_groups = []
        for group in np.unique(protected_attr):
            group_mask = protected_attr == group
            group_positive_rate = np.mean(y[group_mask])
            overall_positive_rate = np.mean(y)
            
            if group_positive_rate < overall_positive_rate * 0.8:  # 阈值
                minority_groups.append(group)
        
        synthetic_samples = []
        synthetic_labels = []
        synthetic_protected = []
        
        for group in minority_groups:
            # 获取该组的正例样本
            group_mask = (protected_attr == group) & (y == 1)
            group_samples = X[group_mask]
            
            if len(group_samples) < k_neighbors:
                continue
            
            # 使用KNN生成合成样本
            nn = NearestNeighbors(n_neighbors=k_neighbors)
            nn.fit(group_samples)
            
            # 为每个样本生成合成样本
            for idx, sample in group_samples.iterrows():
                distances, indices = nn.kneighbors([sample.values])
                
                # 随机选择一个邻居
                neighbor_idx = np.random.choice(indices[0][1:])  # 排除自己
                neighbor = group_samples.iloc[neighbor_idx]
                
                # 生成合成样本
                alpha = np.random.random()
                synthetic_sample = sample + alpha * (neighbor - sample)
                
                synthetic_samples.append(synthetic_sample.values)
                synthetic_labels.append(1)
                synthetic_protected.append(group)
        
        if synthetic_samples:
            synthetic_X = pd.DataFrame(
                synthetic_samples, 
                columns=X.columns
            )
            synthetic_y = np.array(synthetic_labels)
            synthetic_prot = np.array(synthetic_protected)
            
            # 合并原始数据和合成数据
            combined_X = pd.concat([X, synthetic_X], ignore_index=True)
            combined_y = np.concatenate([y, synthetic_y])
            combined_prot = np.concatenate([protected_attr, synthetic_prot])
            
            return combined_X, combined_y, combined_prot
        
        return X, y, protected_attr

# 使用示例
preprocessor = BiasPreprocessor()

# 1. 重新加权
weights = preprocessor.reweighting(X, y, df['gender'].values)
print(f"重新加权完成，权重范围: {weights.min():.3f} - {weights.max():.3f}")

# 2. 差异影响消除
X_repaired = preprocessor.disparate_impact_remover(
    df[['education_score', 'experience_years', 'interview_score', 'gender']], 
    'gender',
    repair_level=0.5
)
print("差异影响消除完成")

# 3. 合成少数群体过采样
X_oversampled, y_oversampled, prot_oversampled = preprocessor.synthetic_minority_oversampling(
    df[['education_score', 'experience_years', 'interview_score']], 
    df['hired'].values,
    df['gender'].values
)
print(f"过采样完成，样本数从 {len(df)} 增加到 {len(X_oversampled)}")
```

#### 处理中方法（In-processing）

```python
class FairClassifier:
    """公平分类器"""
    
    def __init__(self, base_classifier, fairness_constraint='demographic_parity'):
        self.base_classifier = base_classifier
        self.fairness_constraint = fairness_constraint
        self.fairness_penalty = 1.0
        self.trained_model = None
    
    def adversarial_debiasing(self, X, y, protected_attr, epochs=100):
        """对抗去偏方法"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        class Classifier(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x)
        
        class Adversary(nn.Module):
            def __init__(self, hidden_dim=32):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, predictions):
                return self.network(predictions)
        
        # 准备数据
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        prot_tensor = torch.FloatTensor(protected_attr).unsqueeze(1)
        
        # 创建模型
        classifier = Classifier(X.shape[1])
        adversary = Adversary()
        
        # 优化器
        clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        adv_optimizer = optim.Adam(adversary.parameters(), lr=0.001)
        
        # 损失函数
        criterion = nn.BCELoss()
        
        # 训练循环
        for epoch in range(epochs):
            # 训练分类器
            clf_optimizer.zero_grad()
            
            predictions = classifier(X_tensor)
            clf_loss = criterion(predictions, y_tensor)
            
            # 对抗损失（希望对手无法从预测中推断受保护属性）
            adv_predictions = adversary(predictions.detach())
            adv_loss = criterion(adv_predictions, prot_tensor)
            
            # 分类器的总损失（分类损失 - 对抗损失）
            total_clf_loss = clf_loss - self.fairness_penalty * adv_loss
            total_clf_loss.backward()
            clf_optimizer.step()
            
            # 训练对手
            adv_optimizer.zero_grad()
            adv_predictions = adversary(predictions.detach())
            adv_loss = criterion(adv_predictions, prot_tensor)
            adv_loss.backward()
            adv_optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}: Clf Loss: {clf_loss.item():.4f}, Adv Loss: {adv_loss.item():.4f}')
        
        self.trained_model = classifier
        return classifier
    
    def fairness_constrained_optimization(self, X, y, protected_attr):
        """公平性约束优化"""
        from sklearn.linear_model import LogisticRegression
        from scipy.optimize import minimize
        
        def objective_function(weights):
            # 使用权重训练模型
            model = LogisticRegression()
            
            # 创建加权样本
            sample_weights = np.abs(weights)
            model.fit(X, y, sample_weight=sample_weights)
            
            # 计算预测
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            # 分类损失
            classification_loss = -model.score(X, y)
            
            # 公平性损失
            fairness_loss = self._compute_fairness_violation(
                y_pred, y_prob, protected_attr
            )
            
            return classification_loss + self.fairness_penalty * fairness_loss
        
        # 初始权重
        initial_weights = np.ones(len(X))
        
        # 优化
        result = minimize(
            objective_function,
            initial_weights,
            method='L-BFGS-B',
            bounds=[(0.1, 10.0)] * len(X)
        )
        
        # 使用最优权重训练最终模型
        optimal_weights = result.x
        final_model = LogisticRegression()
        final_model.fit(X, y, sample_weight=optimal_weights)
        
        self.trained_model = final_model
        return final_model
    
    def _compute_fairness_violation(self, y_pred, y_prob, protected_attr):
        """计算公平性违反程度"""
        if self.fairness_constraint == 'demographic_parity':
            # 统计均等性违反
            groups = np.unique(protected_attr)
            positive_rates = []
            
            for group in groups:
                group_mask = protected_attr == group
                positive_rate = np.mean(y_pred[group_mask])
                positive_rates.append(positive_rate)
            
            return np.var(positive_rates)  # 方差作为违反度量
        
        elif self.fairness_constraint == 'equalized_odds':
            # 机会均等性违反
            # 这里简化实现，实际需要真实标签
            return 0  # 占位符
        
        return 0
    
    def predict(self, X):
        """预测"""
        if self.trained_model is None:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.trained_model, 'predict'):
            return self.trained_model.predict(X)
        else:
            # PyTorch模型
            import torch
            X_tensor = torch.FloatTensor(X.values)
            with torch.no_grad():
                predictions = self.trained_model(X_tensor)
                return (predictions > 0.5).numpy().astype(int).flatten()

# 使用示例
fair_clf = FairClassifier(
    base_classifier=LogisticRegression(),
    fairness_constraint='demographic_parity'
)

# 设置公平性惩罚权重
fair_clf.fairness_penalty = 0.5

# 训练公平分类器
print("训练对抗去偏模型...")
model = fair_clf.adversarial_debiasing(
    X[['education_score', 'experience_years', 'interview_score']], 
    df['hired'], 
    (df['gender'] == 'F').astype(int).values,
    epochs=50
)

# 预测
y_pred_fair = fair_clf.predict(X[['education_score', 'experience_years', 'interview_score']])

# 比较公平性
print("\n公平性比较:")
original_bias = bias_detector.detect_statistical_parity(
    y_pred, (df['gender'] == 'F').astype(int).values
)
fair_bias = bias_detector.detect_statistical_parity(
    y_pred_fair, (df['gender'] == 'F').astype(int).values
)

print(f"原始模型偏见: {original_bias['max_difference']:.3f}")
print(f"公平模型偏见: {fair_bias['max_difference']:.3f}")
print(f"偏见减少: {(original_bias['max_difference'] - fair_bias['max_difference']):.3f}")
```

#### 后处理方法

```python
class BiasPostprocessor:
    """偏见后处理器"""
    
    def __init__(self):
        self.calibration_thresholds = {}
        self.equalized_odds_thresholds = {}
    
    def calibrated_equalized_odds(self, y_true, y_prob, protected_attr, cost_constraint=None):
        """校准机会均等后处理"""
        groups = np.unique(protected_attr)
        
        # 为每个组找到最优阈值
        optimal_thresholds = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            # 搜索最优阈值
            best_threshold = 0.5
            best_score = float('-inf')
            
            for threshold in np.arange(0.1, 0.9, 0.01):
                group_y_pred = (group_y_prob >= threshold).astype(int)
                
                # 计算性能指标
                if len(np.unique(group_y_true)) > 1 and len(np.unique(group_y_pred)) > 1:
                    tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    # 目标：最大化TPR，最小化FPR
                    score = tpr - fpr
                    
                    if cost_constraint:
                        # 考虑成本约束
                        cost = cost_constraint.get(group, 1.0)
                        score = score / cost
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            optimal_thresholds[group] = best_threshold
        
        self.equalized_odds_thresholds = optimal_thresholds
        
        # 应用阈值生成最终预测
        y_pred_calibrated = np.zeros_like(y_true)
        
        for group in groups:
            group_mask = protected_attr == group
            threshold = optimal_thresholds[group]
            y_pred_calibrated[group_mask] = (y_prob[group_mask] >= threshold).astype(int)
        
        return y_pred_calibrated, optimal_thresholds
    
    def reject_option_classification(self, y_prob, protected_attr, 
                                   critical_region_lower=0.45, 
                                   critical_region_upper=0.55):
        """拒绝选项分类"""
        y_pred_modified = np.zeros(len(y_prob))
        
        # 识别临界区域
        critical_region = (
            (y_prob >= critical_region_lower) & 
            (y_prob <= critical_region_upper)
        )
        
        # 非临界区域使用原始预测
        y_pred_modified[~critical_region] = (y_prob[~critical_region] > 0.5).astype(int)
        
        # 临界区域进行公平性调整
        groups = np.unique(protected_attr)
        
        # 计算各组在临界区域的分布
        group_counts = {}
        for group in groups:
            group_mask = (protected_attr == group) & critical_region
            group_counts[group] = np.sum(group_mask)
        
        # 计算期望的正例分配
        total_critical = np.sum(critical_region)
        overall_positive_rate = 0.5  # 在临界区域假设50%正例率
        
        for group in groups:
            group_mask = (protected_attr == group) & critical_region
            group_size = np.sum(group_mask)
            
            if group_size > 0:
                # 随机分配以达到期望的正例率
                expected_positives = int(group_size * overall_positive_rate)
                
                # 随机选择正例
                group_indices = np.where(group_mask)[0]
                positive_indices = np.random.choice(
                    group_indices, 
                    size=min(expected_positives, len(group_indices)), 
                    replace=False
                )
                
                y_pred_modified[positive_indices] = 1
        
        return y_pred_modified
    
    def multicalibration(self, y_true, y_prob, protected_attr, num_rounds=10):
        """多重校准"""
        calibrated_probs = y_prob.copy()
        
        for round_num in range(num_rounds):
            # 为每个受保护组进行校准
            groups = np.unique(protected_attr)
            
            for group in groups:
                group_mask = protected_attr == group
                group_y_true = y_true[group_mask]
                group_probs = calibrated_probs[group_mask]
                
                # 计算校准误差
                calibration_error = np.mean(group_y_true) - np.mean(group_probs)
                
                # 调整概率
                adjustment = 0.1 * calibration_error  # 学习率
                calibrated_probs[group_mask] += adjustment
                
                # 确保概率在[0,1]范围内
                calibrated_probs[group_mask] = np.clip(
                    calibrated_probs[group_mask], 0, 1
                )
        
        return calibrated_probs
    
    def demographic_parity_postprocessing(self, y_pred, protected_attr, target_rate=None):
        """统计均等性后处理"""
        groups = np.unique(protected_attr)
        
        if target_rate is None:
            # 使用全局正例率作为目标
            target_rate = np.mean(y_pred)
        
        y_pred_adjusted = y_pred.copy()
        
        for group in groups:
            group_mask = protected_attr == group
            group_predictions = y_pred[group_mask]
            current_rate = np.mean(group_predictions)
            
            if current_rate < target_rate:
                # 需要增加正例
                deficit = target_rate - current_rate
                group_size = np.sum(group_mask)
                num_to_flip = int(deficit * group_size)
                
                # 随机选择负例变为正例
                negative_indices = np.where(group_mask & (y_pred == 0))[0]
                if len(negative_indices) >= num_to_flip:
                    flip_indices = np.random.choice(
                        negative_indices, 
                        size=num_to_flip, 
                        replace=False
                    )
                    y_pred_adjusted[flip_indices] = 1
            
            elif current_rate > target_rate:
                # 需要减少正例
                excess = current_rate - target_rate
                group_size = np.sum(group_mask)
                num_to_flip = int(excess * group_size)
                
                # 随机选择正例变为负例
                positive_indices = np.where(group_mask & (y_pred == 1))[0]
                if len(positive_indices) >= num_to_flip:
                    flip_indices = np.random.choice(
                        positive_indices, 
                        size=num_to_flip, 
                        replace=False
                    )
                    y_pred_adjusted[flip_indices] = 0
        
        return y_pred_adjusted

# 使用示例
postprocessor = BiasPostprocessor()

# 获取原始模型的概率预测
from sklearn.ensemble import RandomForestClassifier
original_model = RandomForestClassifier(random_state=42)
original_model.fit(X_train, y_train)
y_prob_original = original_model.predict_proba(X_test)[:, 1]
y_pred_original = original_model.predict(X_test)

# 1. 校准机会均等
protected_test = X_test['gender_F'].values
y_pred_calibrated, thresholds = postprocessor.calibrated_equalized_odds(
    y_test, y_prob_original, protected_test
)

print("校准机会均等阈值:")
for group, threshold in thresholds.items():
    print(f"  组 {group}: {threshold:.3f}")

# 2. 拒绝选项分类
y_pred_reject = postprocessor.reject_option_classification(
    y_prob_original, protected_test
)

# 3. 统计均等性后处理
y_pred_demographic = postprocessor.demographic_parity_postprocessing(
    y_pred_original, protected_test
)

# 比较不同后处理方法的效果
methods = {
    '原始': y_pred_original,
    '校准机会均等': y_pred_calibrated,
    '拒绝选项': y_pred_reject,
    '统计均等性': y_pred_demographic
}

print("\n不同后处理方法的公平性比较:")
for method_name, predictions in methods.items():
    bias_result = bias_detector.detect_statistical_parity(
        predictions, protected_test
    )
    print(f"{method_name}: 最大偏见差异 = {bias_result['max_difference']:.3f}")
```

## 第三节：AI安全与对抗攻击

### 3.1 对抗样本攻击

#### 对抗样本生成

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class AdversarialAttacks:
    """对抗攻击实现"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def fgsm_attack(self, images, labels, epsilon=0.1):
        """快速梯度符号方法攻击"""
        images = images.clone().detach().requires_grad_(True)
        
        # 前向传播
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播获取梯度
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_images = images + epsilon * sign_data_grad
        
        # 确保像素值在有效范围内
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images
    
    def pgd_attack(self, images, labels, epsilon=0.1, alpha=0.01, num_iter=10):
        """投影梯度下降攻击"""
        original_images = images.clone().detach()
        
        # 随机初始化扰动
        perturbed_images = images.clone().detach()
        perturbed_images += torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        for i in range(num_iter):
            perturbed_images.requires_grad_()
            
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # 更新扰动
            data_grad = perturbed_images.grad.data
            perturbed_images = perturbed_images.detach() + alpha * data_grad.sign()
            
            # 投影到epsilon球内
            eta = torch.clamp(perturbed_images - original_images, -epsilon, epsilon)
            perturbed_images = torch.clamp(original_images + eta, 0, 1)
        
        return perturbed_images
    
    def c_w_attack(self, images, labels, c=1.0, kappa=0, max_iter=1000, learning_rate=0.01):
        """Carlini & Wagner攻击"""
        batch_size = images.shape[0]
        
        # 将图像转换到tanh空间
        def to_tanh_space(x):
            return torch.atanh((x - 0.5) * 1.99999)
        
        def from_tanh_space(x):
            return torch.tanh(x) * 0.5 + 0.5
        
        # 初始化扰动变量
        w = to_tanh_space(images.clone()).detach().requires_grad_(True)
        optimizer = optim.Adam([w], lr=learning_rate)
        
        best_adv_images = images.clone()
        best_l2_dist = float('inf') * torch.ones(batch_size)
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # 生成对抗样本
            adv_images = from_tanh_space(w)
            
            # 计算L2距离
            l2_dist = torch.norm((adv_images - images).view(batch_size, -1), dim=1)
            
            # 模型输出
            outputs = self.model(adv_images)
            
            # C&W损失函数
            real_logits = torch.gather(outputs, 1, labels.unsqueeze(1)).squeeze()
            other_logits = torch.max(
                outputs - 1000 * F.one_hot(labels, outputs.shape[1]), 
                dim=1
            )[0]
            
            f_loss = torch.clamp(real_logits - other_logits + kappa, min=0)
            
            # 总损失
            loss = torch.sum(l2_dist) + c * torch.sum(f_loss)
            
            loss.backward()
            optimizer.step()
            
            # 更新最佳对抗样本
            pred_labels = torch.argmax(outputs, dim=1)
            successful_attacks = (pred_labels != labels)
            
            for i in range(batch_size):
                if successful_attacks[i] and l2_dist[i] < best_l2_dist[i]:
                    best_l2_dist[i] = l2_dist[i]
                    best_adv_images[i] = adv_images[i]
        
        return best_adv_images
    
    def evaluate_robustness(self, test_loader, attack_methods, epsilons):
        """评估模型鲁棒性"""
        results = {}
        
        for attack_name, attack_func in attack_methods.items():
            results[attack_name] = {}
            
            for epsilon in epsilons:
                correct = 0
                total = 0
                
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # 生成对抗样本
                    if attack_name == 'FGSM':
                        adv_images = self.fgsm_attack(images, labels, epsilon)
                    elif attack_name == 'PGD':
                        adv_images = self.pgd_attack(images, labels, epsilon)
                    else:
                        continue
                    
                    # 评估对抗样本
                    with torch.no_grad():
                        outputs = self.model(adv_images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                results[attack_name][epsilon] = accuracy
        
        return results

# 使用示例
class SimpleNet(nn.Module):
    """简单的神经网络用于演示"""
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 创建模型和攻击器
model = SimpleNet()
attacker = AdversarialAttacks(model)

# 生成示例数据
batch_size = 32
images = torch.randn(batch_size, 1, 28, 28)
labels = torch.randint(0, 10, (batch_size,))

# 执行不同攻击
print("执行对抗攻击...")
fgsm_adv = attacker.fgsm_attack(images, labels, epsilon=0.1)
pgd_adv = attacker.pgd_attack(images, labels, epsilon=0.1)

print(f"原始图像形状: {images.shape}")
print(f"FGSM对抗样本形状: {fgsm_adv.shape}")
print(f"PGD对抗样本形状: {pgd_adv.shape}")

# 计算扰动大小
fgsm_perturbation = torch.norm((fgsm_adv - images).view(batch_size, -1), dim=1)
pgd_perturbation = torch.norm((pgd_adv - images).view(batch_size, -1), dim=1)

print(f"FGSM平均扰动L2范数: {fgsm_perturbation.mean():.4f}")
print(f"PGD平均扰动L2范数: {pgd_perturbation.mean():.4f}")
```

### 3.2 防御机制

#### 对抗训练

```python
class AdversarialTraining:
    """对抗训练实现"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.attacker = AdversarialAttacks(model, device)
    
    def adversarial_training(self, train_loader, num_epochs=10, 
                           epsilon=0.1, alpha=0.01, attack_iters=7):
        """对抗训练主循环"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 生成对抗样本
                adv_data = self.attacker.pgd_attack(
                    data, target, epsilon, alpha, attack_iters
                )
                
                # 混合训练：50%原始样本 + 50%对抗样本
                mixed_data = torch.cat([data, adv_data], dim=0)
                mixed_target = torch.cat([target, target], dim=0)
                
                optimizer.zero_grad()
                output = self.model(mixed_data)
                loss = criterion(output, mixed_target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(mixed_target.view_as(pred)).sum().item()
                total += mixed_target.size(0)
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
        
        return self.model
    
    def trades_training(self, train_loader, num_epochs=10, 
                       beta=6.0, epsilon=0.1, step_size=0.01, num_steps=10):
        """TRADES训练方法"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        def trades_loss(model, x_natural, y, beta, step_size, epsilon, num_steps):
            # 自然损失
            logits = model(x_natural)
            natural_loss = F.cross_entropy(logits, y)
            
            # 生成对抗样本
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(self.device)
            
            for _ in range(num_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = F.kl_div(
                        F.log_softmax(model(x_adv), dim=1),
                        F.softmax(model(x_natural), dim=1),
                        reduction='sum'
                    )
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
            # 鲁棒性损失
            robust_loss = F.kl_div(
                F.log_softmax(model(x_adv), dim=1),
                F.softmax(model(x_natural), dim=1),
                reduction='batchmean'
            )
            
            return natural_loss + beta * robust_loss
        
        self.model.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                loss = trades_loss(self.model, data, target, beta, step_size, epsilon, num_steps)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'TRADES Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        return self.model
```

#### 检测机制

```python
class AdversarialDetector:
    """对抗样本检测器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.detection_thresholds = {}
    
    def statistical_detection(self, images, clean_stats):
        """基于统计特征的检测"""
        batch_stats = {
            'mean': torch.mean(images, dim=(2, 3)),
            'std': torch.std(images, dim=(2, 3)),
            'min': torch.min(images.view(images.size(0), -1), dim=1)[0],
            'max': torch.max(images.view(images.size(0), -1), dim=1)[0]
        }
        
        anomaly_scores = []
        
        for i in range(images.size(0)):
            score = 0
            for stat_name, stat_values in batch_stats.items():
                if stat_name in clean_stats:
                    # 计算与正常统计的偏差
                    deviation = abs(stat_values[i] - clean_stats[stat_name]['mean'])
                    normalized_deviation = deviation / clean_stats[stat_name]['std']
                    score += normalized_deviation.sum().item()
            
            anomaly_scores.append(score)
        
        return np.array(anomaly_scores)
    
    def prediction_inconsistency_detection(self, images, num_transforms=10):
        """基于预测不一致性的检测"""
        self.model.eval()
        
        # 定义随机变换
        transforms_list = [
            lambda x: x + 0.01 * torch.randn_like(x),  # 添加噪声
            lambda x: F.interpolate(F.interpolate(x, scale_factor=0.9), size=x.shape[2:]),  # 缩放
            lambda x: torch.roll(x, shifts=1, dims=2),  # 平移
        ]
        
        original_predictions = []
        transformed_predictions = []
        
        with torch.no_grad():
            # 原始预测
            original_output = self.model(images)
            original_pred = F.softmax(original_output, dim=1)
            
            # 变换后预测
            for _ in range(num_transforms):
                transform = np.random.choice(transforms_list)
                transformed_images = transform(images.clone())
                transformed_output = self.model(transformed_images)
                transformed_pred = F.softmax(transformed_output, dim=1)
                transformed_predictions.append(transformed_pred)
        
        # 计算预测不一致性
        inconsistency_scores = []
        
        for i in range(images.size(0)):
            original_prob = original_pred[i]
            
            # 计算与变换预测的差异
            total_diff = 0
            for transformed_pred in transformed_predictions:
                diff = torch.norm(original_prob - transformed_pred[i], p=2)
                total_diff += diff.item()
            
            avg_inconsistency = total_diff / num_transforms
            inconsistency_scores.append(avg_inconsistency)
        
        return np.array(inconsistency_scores)
    
    def neural_network_detection(self, images, detector_model):
        """基于神经网络的检测"""
        detector_model.eval()
        
        with torch.no_grad():
            # 提取特征
            features = self._extract_features(images)
            
            # 检测器预测
            detection_scores = detector_model(features)
            detection_probs = torch.sigmoid(detection_scores)
        
        return detection_probs.cpu().numpy().flatten()
    
    def _extract_features(self, images):
        """提取用于检测的特征"""
        # 这里可以使用预训练模型的中间层特征
        # 简化实现：使用图像的统计特征
        batch_size = images.size(0)
        features = []
        
        for i in range(batch_size):
            img = images[i]
            
            # 统计特征
            mean_val = torch.mean(img)
            std_val = torch.std(img)
            min_val = torch.min(img)
            max_val = torch.max(img)
            
            # 梯度特征
            grad_x = torch.diff(img, dim=1)
            grad_y = torch.diff(img, dim=2)
            grad_magnitude = torch.sqrt(grad_x[:, :, :-1]**2 + grad_y[:, :-1, :]**2)
            avg_grad = torch.mean(grad_magnitude)
            
            feature_vector = torch.tensor([
                mean_val, std_val, min_val, max_val, avg_grad
            ])
            features.append(feature_vector)
        
        return torch.stack(features)
    
    def ensemble_detection(self, images, clean_stats=None, detector_model=None):
        """集成检测方法"""
        detection_scores = {}
        
        # 统计检测
        if clean_stats is not None:
            stat_scores = self.statistical_detection(images, clean_stats)
            detection_scores['statistical'] = stat_scores
        
        # 不一致性检测
        inconsistency_scores = self.prediction_inconsistency_detection(images)
        detection_scores['inconsistency'] = inconsistency_scores
        
        # 神经网络检测
        if detector_model is not None:
            nn_scores = self.neural_network_detection(images, detector_model)
            detection_scores['neural_network'] = nn_scores
        
        # 集成分数
        ensemble_scores = np.zeros(images.size(0))
        weights = {'statistical': 0.3, 'inconsistency': 0.4, 'neural_network': 0.3}
        
        for method, scores in detection_scores.items():
            if method in weights:
                # 归一化分数到[0,1]
                normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
                ensemble_scores += weights[method] * normalized_scores
        
        return ensemble_scores, detection_scores

# 使用示例
detector = AdversarialDetector(model)

# 准备正常样本统计
clean_images = torch.randn(100, 1, 28, 28)
clean_stats = {
    'mean': {
        'mean': torch.mean(clean_images, dim=(2, 3)).mean(),
        'std': torch.mean(clean_images, dim=(2, 3)).std()
    },
    'std': {
        'mean': torch.std(clean_images, dim=(2, 3)).mean(),
        'std': torch.std(clean_images, dim=(2, 3)).std()
    }
}

# 检测对抗样本
test_images = torch.randn(10, 1, 28, 28)
ensemble_scores, individual_scores = detector.ensemble_detection(
    test_images, 
    clean_stats=clean_stats
)

print("对抗样本检测结果:")
for i, score in enumerate(ensemble_scores):
    print(f"样本 {i}: 异常分数 = {score:.3f}")

# 设置检测阈值
threshold = 0.5
detected_adversarial = ensemble_scores > threshold
print(f"\n检测到 {np.sum(detected_adversarial)} 个可疑样本")
```

## 第四节：隐私保护与数据安全

### 4.1 差分隐私

#### 差分隐私基础

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import hashlib

class DifferentialPrivacy:
    """差分隐私实现"""
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon  # 隐私预算
        self.delta = delta      # 失败概率
    
    def laplace_mechanism(self, true_answer, sensitivity):
        """拉普拉斯机制"""
        # 计算噪声尺度
        scale = sensitivity / self.epsilon
        
        # 添加拉普拉斯噪声
        noise = np.random.laplace(0, scale)
        noisy_answer = true_answer + noise
        
        return noisy_answer
    
    def gaussian_mechanism(self, true_answer, sensitivity, delta=None):
        """高斯机制"""
        if delta is None:
            delta = self.delta
        
        # 计算噪声标准差
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        
        # 添加高斯噪声
        noise = np.random.normal(0, sigma)
        noisy_answer = true_answer + noise
        
        return noisy_answer
    
    def exponential_mechanism(self, candidates, utility_function, sensitivity):
        """指数机制"""
        # 计算每个候选的效用
        utilities = [utility_function(candidate) for candidate in candidates]
        
        # 计算概率权重
        weights = np.exp(self.epsilon * np.array(utilities) / (2 * sensitivity))
        probabilities = weights / np.sum(weights)
        
        # 根据概率选择候选
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]
    
    def composition_privacy_loss(self, mechanisms):
        """计算组合隐私损失"""
        # 简单组合
        total_epsilon = sum(mech['epsilon'] for mech in mechanisms)
        total_delta = sum(mech['delta'] for mech in mechanisms)
        
        # 高级组合（近似）
        k = len(mechanisms)
        if k > 1:
            advanced_epsilon = np.sqrt(2 * k * np.log(1/self.delta)) * self.epsilon + k * self.epsilon * (np.exp(self.epsilon) - 1)
            return {
                'simple_composition': {'epsilon': total_epsilon, 'delta': total_delta},
                'advanced_composition': {'epsilon': advanced_epsilon, 'delta': self.delta}
            }
        
        return {'simple_composition': {'epsilon': total_epsilon, 'delta': total_delta}}

class PrivateDataAnalysis:
    """私有数据分析"""
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.dp = DifferentialPrivacy(epsilon, delta)
    
    def private_mean(self, data, bounds):
        """差分隐私均值计算"""
        # 裁剪数据到指定范围
        clipped_data = np.clip(data, bounds[0], bounds[1])
        
        # 计算真实均值
        true_mean = np.mean(clipped_data)
        
        # 敏感度计算
        sensitivity = (bounds[1] - bounds[0]) / len(data)
        
        # 添加噪声
        private_mean = self.dp.laplace_mechanism(true_mean, sensitivity)
        
        return private_mean
    
    def private_histogram(self, data, bins):
        """差分隐私直方图"""
        # 计算真实直方图
        hist, _ = np.histogram(data, bins=bins)
        
        # 敏感度为1（添加或删除一个数据点最多改变一个bin的计数）
        sensitivity = 1
        
        # 为每个bin添加噪声
        private_hist = []
        for count in hist:
            noisy_count = self.dp.laplace_mechanism(count, sensitivity)
            # 确保计数非负
            private_hist.append(max(0, noisy_count))
        
        return np.array(private_hist)
    
    def private_covariance(self, data1, data2, bounds1, bounds2):
        """差分隐私协方差计算"""
        # 裁剪数据
        clipped_data1 = np.clip(data1, bounds1[0], bounds1[1])
        clipped_data2 = np.clip(data2, bounds2[0], bounds2[1])
        
        # 计算真实协方差
        true_cov = np.cov(clipped_data1, clipped_data2)[0, 1]
        
        # 敏感度计算
        range1 = bounds1[1] - bounds1[0]
        range2 = bounds2[1] - bounds2[0]
        sensitivity = range1 * range2 / len(data1)
        
        # 添加噪声
        private_cov = self.dp.laplace_mechanism(true_cov, sensitivity)
        
        return private_cov

# 使用示例
np.random.seed(42)

# 生成示例数据
data = np.random.normal(50, 15, 1000)
data2 = data + np.random.normal(0, 5, 1000)

# 创建私有数据分析器
private_analyzer = PrivateDataAnalysis(epsilon=1.0)

# 计算私有统计量
true_mean = np.mean(data)
private_mean = private_analyzer.private_mean(data, bounds=[0, 100])

print(f"真实均值: {true_mean:.2f}")
print(f"差分隐私均值: {private_mean:.2f}")
print(f"误差: {abs(true_mean - private_mean):.2f}")

# 私有直方图
bins = np.linspace(0, 100, 21)
true_hist, _ = np.histogram(data, bins=bins)
private_hist = private_analyzer.private_histogram(data, bins)

print(f"\n直方图比较（前5个bin）:")
for i in range(5):
    print(f"Bin {i}: 真实={true_hist[i]}, 私有={private_hist[i]:.1f}")

# 私有协方差
true_cov = np.cov(data, data2)[0, 1]
private_cov = private_analyzer.private_covariance(data, data2, [0, 100], [0, 100])

print(f"\n真实协方差: {true_cov:.2f}")
print(f"差分隐私协方差: {private_cov:.2f}")
```

### 4.2 联邦学习

#### 联邦学习框架

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

class FederatedLearning:
    """联邦学习实现"""
    
    def __init__(self, global_model, num_clients, learning_rate=0.01):
        self.global_model = global_model
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.client_models = []
        
        # 为每个客户端创建模型副本
        for _ in range(num_clients):
            client_model = copy.deepcopy(global_model)
            self.client_models.append(client_model)
    
    def client_update(self, client_id, train_loader, num_epochs=1):
        """客户端本地更新"""
        model = self.client_models[client_id]
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
    def federated_averaging(self, client_weights, client_sizes):
        """联邦平均算法"""
        # 计算加权平均
        total_size = sum(client_sizes)
        averaged_weights = {}
        
        # 初始化平均权重
        for key in client_weights[0].keys():
            averaged_weights[key] = torch.zeros_like(client_weights[0][key])
        
        # 加权求和
        for i, weights in enumerate(client_weights):
            weight = client_sizes[i] / total_size
            for key in weights.keys():
                averaged_weights[key] += weight * weights[key]
        
        return averaged_weights
    
    def train_round(self, client_data_loaders, client_sizes):
        """一轮联邦训练"""
        client_weights = []
        
        # 客户端本地训练
        for client_id in range(self.num_clients):
            # 同步全局模型到客户端
            self.client_models[client_id].load_state_dict(self.global_model.state_dict())
            
            # 本地训练
            weights = self.client_update(client_id, client_data_loaders[client_id])
            client_weights.append(weights)
        
        # 聚合更新
        averaged_weights = self.federated_averaging(client_weights, client_sizes)
        
        # 更新全局模型
        self.global_model.load_state_dict(averaged_weights)
        
        return averaged_weights
    
    def evaluate_global_model(self, test_loader):
        """评估全局模型"""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy

class SecureAggregation:
    """安全聚合协议"""
    
    def __init__(self, num_clients, threshold):
        self.num_clients = num_clients
        self.threshold = threshold  # 最少需要的客户端数量
    
    def generate_secret_shares(self, secret, num_shares, threshold):
        """生成秘密分享"""
        # 简化的Shamir秘密分享实现
        coefficients = [secret] + [np.random.randint(0, 1000) for _ in range(threshold - 1)]
        
        shares = []
        for i in range(1, num_shares + 1):
            share_value = sum(coef * (i ** j) for j, coef in enumerate(coefficients)) % 1009  # 使用质数模
            shares.append((i, share_value))
        
        return shares
    
    def reconstruct_secret(self, shares):
        """重构秘密"""
        # 拉格朗日插值重构
        def lagrange_interpolation(shares, x=0):
            result = 0
            for i, (xi, yi) in enumerate(shares):
                term = yi
                for j, (xj, _) in enumerate(shares):
                    if i != j:
                        term = (term * (x - xj) * pow(xi - xj, -1, 1009)) % 1009
                result = (result + term) % 1009
            return result
        
        return lagrange_interpolation(shares)
    
    def secure_aggregate(self, client_updates):
        """安全聚合客户端更新"""
        # 简化实现：假设所有客户端都参与
        aggregated_update = {}
        
        for key in client_updates[0].keys():
            # 对每个参数进行安全聚合
            param_sum = torch.zeros_like(client_updates[0][key])
            
            for client_update in client_updates:
                param_sum += client_update[key]
            
            # 计算平均值
            aggregated_update[key] = param_sum / len(client_updates)
        
        return aggregated_update

# 使用示例
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建全局模型
global_model = SimpleNet()

# 模拟客户端数据
num_clients = 5
client_data = []
client_sizes = []

for i in range(num_clients):
    # 生成非IID数据
    size = np.random.randint(100, 500)
    data = torch.randn(size, 784)
    labels = torch.randint(0, 10, (size,))
    
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    client_data.append(dataloader)
    client_sizes.append(size)

# 创建联邦学习系统
fed_system = FederatedLearning(global_model, num_clients)

# 训练多轮
num_rounds = 10
for round_num in range(num_rounds):
    print(f"\n联邦学习轮次 {round_num + 1}")
    
    # 执行一轮训练
    fed_system.train_round(client_data, client_sizes)
    
    # 评估（使用第一个客户端的数据作为测试集）
    accuracy = fed_system.evaluate_global_model(client_data[0])
    print(f"全局模型准确率: {accuracy:.2f}%")

print("\n联邦学习训练完成")
```

## 第五节：可解释性AI

### 5.1 模型解释方法

#### LIME (Local Interpretable Model-agnostic Explanations)

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class LIME:
    """LIME解释器实现"""
    
    def __init__(self, model, mode='classification'):
        self.model = model
        self.mode = mode
    
    def explain_instance(self, instance, num_features=10, num_samples=1000):
        """解释单个实例"""
        # 生成扰动样本
        perturbed_samples = self._generate_samples(instance, num_samples)
        
        # 获取模型预测
        predictions = self._get_predictions(perturbed_samples)
        
        # 计算距离权重
        distances = self._compute_distances(instance, perturbed_samples)
        weights = self._compute_weights(distances)
        
        # 训练局部线性模型
        local_model = self._train_local_model(perturbed_samples, predictions, weights)
        
        # 获取特征重要性
        feature_importance = self._get_feature_importance(local_model, num_features)
        
        return feature_importance
    
    def _generate_samples(self, instance, num_samples):
        """生成扰动样本"""
        samples = []
        
        for _ in range(num_samples):
            # 随机选择特征进行扰动
            sample = instance.copy()
            
            # 对于图像数据，随机遮挡一些像素
            if len(instance.shape) > 1:  # 图像数据
                mask = np.random.binomial(1, 0.5, instance.shape)
                sample = sample * mask
            else:  # 表格数据
                # 随机替换一些特征值
                num_features_to_change = np.random.randint(1, len(instance) // 2)
                indices = np.random.choice(len(instance), num_features_to_change, replace=False)
                
                for idx in indices:
                    # 用随机值或均值替换
                    sample[idx] = np.random.normal(np.mean(instance), np.std(instance))
            
            samples.append(sample)
        
        return np.array(samples)
    
    def _get_predictions(self, samples):
        """获取模型预测"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for sample in samples:
                if isinstance(sample, np.ndarray):
                    sample_tensor = torch.FloatTensor(sample).unsqueeze(0)
                else:
                    sample_tensor = sample.unsqueeze(0)
                
                output = self.model(sample_tensor)
                
                if self.mode == 'classification':
                    pred = torch.softmax(output, dim=1).numpy()[0]
                else:
                    pred = output.numpy()[0]
                
                predictions.append(pred)
        
        return np.array(predictions)
    
    def _compute_distances(self, instance, samples):
        """计算样本与原实例的距离"""
        distances = []
        
        for sample in samples:
            # 使用欧几里得距离
            distance = np.linalg.norm(instance.flatten() - sample.flatten())
            distances.append(distance)
        
        return np.array(distances)
    
    def _compute_weights(self, distances, kernel_width=0.25):
        """计算样本权重"""
        # 使用指数核函数
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        return weights
    
    def _train_local_model(self, samples, predictions, weights):
        """训练局部线性模型"""
        # 将样本展平
        X = samples.reshape(samples.shape[0], -1)
        
        if self.mode == 'classification':
            # 对于分类，使用最高概率类别
            y = np.argmax(predictions, axis=1)
        else:
            y = predictions.flatten()
        
        # 训练加权线性回归
        local_model = LinearRegression()
        local_model.fit(X, y, sample_weight=weights)
        
        return local_model
    
    def _get_feature_importance(self, local_model, num_features):
        """获取特征重要性"""
        coefficients = local_model.coef_
        
        # 获取最重要的特征
        feature_importance = np.abs(coefficients)
        top_indices = np.argsort(feature_importance)[-num_features:]
        
        importance_dict = {}
        for idx in top_indices:
            importance_dict[idx] = coefficients[idx]
        
        return importance_dict

class SHAP:
    """SHAP值计算器"""
    
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data
    
    def explain_instance(self, instance, num_samples=100):
        """计算实例的SHAP值"""
        num_features = len(instance.flatten())
        shap_values = np.zeros(num_features)
        
        # 计算所有特征子集的边际贡献
        for i in range(num_features):
            marginal_contributions = []
            
            # 采样不同的特征子集
            for _ in range(num_samples):
                # 随机选择特征子集
                subset_size = np.random.randint(0, num_features)
                subset = np.random.choice(num_features, subset_size, replace=False)
                
                # 计算包含和不包含当前特征的预测差异
                with_feature = self._predict_with_subset(instance, np.append(subset, i))
                without_feature = self._predict_with_subset(instance, subset)
                
                marginal_contribution = with_feature - without_feature
                marginal_contributions.append(marginal_contribution)
            
            # 平均边际贡献作为SHAP值
            shap_values[i] = np.mean(marginal_contributions)
        
        return shap_values
    
    def _predict_with_subset(self, instance, feature_subset):
        """使用特征子集进行预测"""
        # 创建混合实例：选中的特征使用原值，其他使用背景值
        mixed_instance = self.background_data.copy()
        
        for feature_idx in feature_subset:
            if len(instance.shape) > 1:  # 图像数据
                coords = np.unravel_index(feature_idx, instance.shape)
                mixed_instance[coords] = instance[coords]
            else:  # 表格数据
                mixed_instance[feature_idx] = instance[feature_idx]
        
        # 获取预测
        self.model.eval()
        with torch.no_grad():
            if isinstance(mixed_instance, np.ndarray):
                input_tensor = torch.FloatTensor(mixed_instance).unsqueeze(0)
            else:
                input_tensor = mixed_instance.unsqueeze(0)
            
            output = self.model(input_tensor)
            prediction = torch.softmax(output, dim=1).numpy()[0, 0]  # 假设二分类
        
        return prediction

class GradCAM:
    """Grad-CAM可视化"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx=None):
        """生成类激活图"""
        self.model.eval()
        
        # 前向传播
        input_image.requires_grad_()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 生成CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # 归一化到[0, 1]
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.squeeze().detach().numpy()

# 使用示例
class ExampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型和示例数据
model = ExampleCNN()
sample_image = torch.randn(1, 1, 28, 28)
sample_tabular = np.random.randn(10)

# LIME解释
lime_explainer = LIME(model)
lime_explanation = lime_explainer.explain_instance(sample_tabular)
print("LIME特征重要性:")
for feature_idx, importance in lime_explanation.items():
    print(f"特征 {feature_idx}: {importance:.4f}")

# Grad-CAM可视化
grad_cam = GradCAM(model, model.conv2)
cam = grad_cam.generate_cam(sample_image)
print(f"\nGrad-CAM热力图形状: {cam.shape}")

# SHAP解释
background = np.random.randn(10)
shap_explainer = SHAP(model, background)
shap_values = shap_explainer.explain_instance(sample_tabular)
print(f"\nSHAP值: {shap_values[:5]}...")  # 显示前5个值
```

## 实践项目

### 项目一：AI伦理评估系统

**目标**: 构建一个综合的AI系统伦理评估框架

**要求**:
1. 实现多维度伦理评估指标
2. 集成偏见检测和公平性分析
3. 提供可解释性分析报告
4. 设计伦理决策支持系统

**核心代码框架**:

```python
class AIEthicsEvaluator:
    """AI伦理评估系统"""
    
    def __init__(self):
        self.bias_detector = BiasDetector()
        self.fairness_analyzer = FairnessAnalyzer()
        self.explainer = LIME(None)  # 模型将在评估时设置
        self.privacy_analyzer = DifferentialPrivacy()
    
    def comprehensive_evaluation(self, model, data, labels, protected_attributes):
        """综合伦理评估"""
        evaluation_report = {
            'bias_analysis': {},
            'fairness_metrics': {},
            'explainability_score': 0,
            'privacy_risk': 0,
            'overall_ethics_score': 0
        }
        
        # 偏见分析
        bias_results = self.bias_detector.comprehensive_bias_analysis(
            data, labels, protected_attributes
        )
        evaluation_report['bias_analysis'] = bias_results
        
        # 公平性分析
        predictions = model.predict(data)
        fairness_results = self.fairness_analyzer.evaluate_fairness(
            predictions, labels, protected_attributes
        )
        evaluation_report['fairness_metrics'] = fairness_results
        
        # 可解释性评估
        explainability_score = self._evaluate_explainability(model, data)
        evaluation_report['explainability_score'] = explainability_score
        
        # 隐私风险评估
        privacy_risk = self._evaluate_privacy_risk(model, data)
        evaluation_report['privacy_risk'] = privacy_risk
        
        # 计算综合伦理分数
        overall_score = self._calculate_overall_ethics_score(evaluation_report)
        evaluation_report['overall_ethics_score'] = overall_score
        
        return evaluation_report
    
    def _evaluate_explainability(self, model, data):
        """评估模型可解释性"""
        # 简化的可解释性评分
        # 实际实现应该更复杂
        return np.random.uniform(0.6, 0.9)
    
    def _evaluate_privacy_risk(self, model, data):
        """评估隐私风险"""
        # 简化的隐私风险评估
        return np.random.uniform(0.1, 0.4)
    
    def _calculate_overall_ethics_score(self, report):
        """计算综合伦理分数"""
        # 权重可以根据应用场景调整
        weights = {
            'fairness': 0.3,
            'explainability': 0.25,
            'privacy': 0.25,
            'bias': 0.2
        }
        
        fairness_score = 1 - report['fairness_metrics'].get('max_difference', 0.5)
        explainability_score = report['explainability_score']
        privacy_score = 1 - report['privacy_risk']
        bias_score = 1 - report['bias_analysis'].get('max_difference', 0.5)
        
        overall_score = (
            weights['fairness'] * fairness_score +
            weights['explainability'] * explainability_score +
            weights['privacy'] * privacy_score +
            weights['bias'] * bias_score
        )
        
        return overall_score
```

### 项目二：对抗鲁棒性测试平台

**目标**: 开发一个全面的AI模型对抗鲁棒性测试平台

**要求**:
1. 集成多种对抗攻击方法
2. 实现自动化鲁棒性评估
3. 提供防御机制建议
4. 生成详细的安全评估报告

### 项目三：隐私保护机器学习系统

**目标**: 构建支持差分隐私和联邦学习的机器学习系统

**要求**:
1. 实现差分隐私训练算法
2. 支持联邦学习协议
3. 集成安全聚合机制
4. 提供隐私预算管理

### 项目四：可解释AI决策支持系统

**目标**: 开发面向特定领域的可解释AI决策支持系统

**要求**:
1. 集成多种解释方法（LIME、SHAP、Grad-CAM等）
2. 提供交互式解释界面
3. 支持反事实解释
4. 生成自然语言解释报告

## 学习评估

### 理论评估

1. **伦理理论理解**（25分）
   - AI伦理基本原则
   - 算法偏见的类型和来源
   - 公平性的不同定义
   - 隐私保护的理论基础

2. **安全技术掌握**（25分）
   - 对抗攻击的原理和方法
   - 防御机制的设计思路
   - 差分隐私的数学基础
   - 联邦学习的协议设计

3. **可解释性方法**（25分）
   - 不同解释方法的适用场景
   - 解释质量的评估标准
   - 可解释性与性能的权衡
   - 解释的可信度分析

4. **综合应用能力**（25分）
   - 伦理问题的识别和分析
   - 技术方案的设计和实现
   - 评估指标的选择和应用
   - 实际问题的解决能力

### 实践评估

1. **项目完成质量**（40分）
   - 代码实现的正确性和效率
   - 系统设计的合理性
   - 功能的完整性和可用性
   - 文档和注释的质量

2. **创新性和深度**（30分）
   - 解决方案的创新性
   - 技术实现的深度
   - 问题分析的全面性
   - 改进和优化的思考

3. **实验设计和分析**（30分）
   - 实验设计的科学性
   - 评估指标的合理性
   - 结果分析的深入性
   - 结论的可靠性

## 延伸学习

### 前沿研究方向

1. **AI治理与政策**
   - AI监管框架研究
   - 算法审计标准
   - 国际AI伦理准则
   - AI责任归属机制

2. **技术伦理前沿**
   - 神经符号AI的伦理考量
   - 量子机器学习的安全性
   - 边缘AI的隐私保护
   - 人机协作的伦理框架

3. **社会影响研究**
   - AI对就业的影响
   - 算法决策的社会公正
   - AI在教育中的伦理问题
   - 医疗AI的伦理挑战

### 推荐资源

**学术期刊**:
- AI & Society
- Ethics and Information Technology
- Journal of AI Research (JAIR)
- Nature Machine Intelligence

**会议和研讨会**:
- FAccT (Fairness, Accountability, and Transparency)
- AIES (AI, Ethics, and Society)
- NeurIPS Workshop on Trustworthy ML
- ICML Workshop on Responsible AI

**在线资源**:
- Partnership on AI
- AI Ethics Lab
- Future of Humanity Institute
- Center for AI Safety

**工具和框架**:
- Fairlearn (公平性工具包)
- AI Fairness 360 (IBM)
- What-If Tool (Google)
- Adversarial Robustness Toolbox

## 模块总结

AI伦理与安全是人工智能发展中不可忽视的重要议题。本模块通过系统性的理论学习和实践训练，帮助学习者:

1. **建立伦理意识**: 理解AI系统可能带来的伦理挑战，培养负责任的AI开发和应用意识

2. **掌握技术方法**: 学习偏见检测、公平性保证、对抗防御、隐私保护等关键技术

3. **提升实践能力**: 通过项目实践，获得解决实际AI伦理和安全问题的经验

4. **培养批判思维**: 能够从多个角度分析AI系统的社会影响，提出建设性的改进建议

随着AI技术的快速发展和广泛应用，AI伦理与安全将成为每个AI从业者必须具备的核心素养。希望通过本模块的学习，能够为构建更加公平、安全、可信的AI系统贡献力量。