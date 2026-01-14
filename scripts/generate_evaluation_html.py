import json
import os

def generate_html():
    # Paths
    base_dir = "/Users/xuke/OtherProject/_self/cg/outputs/2025部门评分"
    self_eval_path = os.path.join(base_dir, "员工自评.json")
    dept_eval_path = os.path.join(base_dir, "2025部门评价汇总.json")
    output_dir = os.path.join(base_dir, "评价对比")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    with open(self_eval_path, 'r', encoding='utf-8') as f:
        self_eval_data = json.load(f)
    
    with open(dept_eval_path, 'r', encoding='utf-8') as f:
        dept_eval_data = json.load(f)

    def get_dev_report(name):
        report_path = os.path.join(base_dir, f"{name}发展报告.md")
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    # Convert dept_eval to dict for easier lookup
    dept_map = {item['姓名']: item for item in dept_eval_data}

    # Radar Dimensions
    dimensions = [
        ("诚信", "诚信"),
        ("责任", "责任"),
        ("协同", "协同"),
        ("创新", "创新"),
        ("工作贡献度", "工作贡献度"),
        ("工作及时性", "工作及时性"),
        ("工作质量", "工作质量"),
        ("学习敏捷度", "潜力-学习敏捷度"),
        ("职业抱负", "潜力-职业抱负"),
        ("敬业度", "潜力-敬业度")
    ]

    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{NAME}} - 2025年度评价对比</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {
            font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: #fff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        header {
            text-align: center;
            border-bottom: 2px solid #eee;
            margin-bottom: 30px;
            padding-bottom: 20px;
        }
        h1 { margin: 0; color: #1a73e8; }
        .role { color: #666; margin-top: 5px; }
        
        #radar-chart {
            width: 100%;
            height: 500px;
            margin-bottom: 40px;
        }
        
        .info-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .card {
            padding: 20px;
            border-radius: 8px;
            background: #f1f3f4;
        }
        .card h3 {
            margin-top: 0;
            color: #1a73e8;
            font-size: 16px;
            border-left: 4px solid #1a73e8;
            padding-left: 10px;
        }
        .full-width {
            grid-column: span 2;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        .meta-item {
            background: #fff;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .meta-label { font-size: 12px; color: #888; display: block; }
        .meta-value { font-weight: bold; font-size: 14px; }
        
        .footer-note {
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{NAME}}</h1>
            <div class="role">{{ROLE}} | 2025年度评价分析报告</div>
        </header>
        
        <div id="radar-chart"></div>
        
        <div class="info-section">
            <div class="card">
                <h3>优势亮点</h3>
                <p>{{STRENGTHS}}</p>
            </div>
            <div class="card">
                <h3>待提升项</h3>
                <p>{{IMPROVEMENTS}}</p>
            </div>
            <div class="card full-width">
                <h3>任用建议</h3>
                <p>{{RECOMMENDATION}}</p>
            </div>
            <div class="card full-width">
                <h3>目前核心工作</h3>
                <p>{{WORK_CONTENT}}</p>
            </div>
            {{DEV_REPORT_SECTION}}
        </div>

        <div class="meta-grid">
            <div class="meta-item">
                <span class="meta-label">红绿灯定位</span>
                <span class="meta-value" style="color: {{LIGHT_COLOR}}">{{LIGHT}}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">潜力等级</span>
                <span class="meta-value">{{POTENTIAL}}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">关联系/继任者</span>
                <span class="meta-value">{{SUCCESSOR}}</span>
            </div>
        </div>
        
        <div class="footer-note">
            * 报告自动生成 | 仅限内部复核使用
        </div>
    </div>

    <script>
        var chartDom = document.getElementById('radar-chart');
        var myChart = echarts.init(chartDom);
        var option;

        option = {
            title: {
                text: '评价模型对比',
                left: 'center',
                top: 0
            },
            legend: {
                data: ['员工自评', '部门评价'],
                bottom: 10
            },
            radar: {
                indicator: {{INDICATORS}},
                shape: 'circle',
                splitNumber: 5,
                axisName: {
                    color: '#888'
                },
                splitLine: {
                    lineStyle: {
                        color: [
                            'rgba(238, 238, 238, 1)',
                            'rgba(238, 238, 238, 0.8)',
                            'rgba(238, 238, 238, 0.6)',
                            'rgba(238, 238, 238, 0.4)',
                            'rgba(238, 238, 238, 0.2)'
                        ].reverse()
                    }
                },
                splitArea: {
                    show: false
                },
                axisLine: {
                    lineStyle: {
                        color: 'rgba(238, 238, 238, 0.5)'
                    }
                }
            },
            series: [
                {
                    name: 'Evaluation Comparison',
                    type: 'radar',
                    data: [
                        {
                            value: {{SELF_DATA}},
                            name: '员工自评',
                            areaStyle: {
                                color: 'rgba(26, 115, 232, 0.2)'
                            },
                            lineStyle: {
                                color: '#1a73e8',
                                width: 2
                            },
                            itemStyle: {
                                color: '#1a73e8'
                            }
                        },
                        {
                            value: {{DEPT_DATA}},
                            name: '部门评价',
                            areaStyle: {
                                color: 'rgba(234, 67, 53, 0.2)'
                            },
                            lineStyle: {
                                color: '#ea4335',
                                width: 2
                            },
                            itemStyle: {
                                color: '#ea4335'
                            }
                        }
                    ]
                }
            ]
        };

        myChart.setOption(option);
    </script>
</body>
</html>
    """

    for item in self_eval_data:
        name = item['填写者']
        if name not in dept_map:
            print(f"Skipping {name}, not found in dept eval.")
            continue
            
        dept_item = dept_map[name]
        
        # Prepare radar data
        self_values = []
        dept_values = []
        indicators = []
        
        for k_self, k_dept in dimensions:
            # Handle indicators
            indicators.append({"name": k_self, "max": 7 if "潜力" not in k_dept else 3})
            
            # Values
            self_values.append(item.get(k_self, 0))
            dept_values.append(dept_item.get(k_dept, 0))
            
        # Replace template
        html = html_template.replace("{{NAME}}", name)
        html = html.replace("{{ROLE}}", dept_item.get("岗位", "未知"))
        html = html.replace("{{STRENGTHS}}", dept_item.get("优点", "无"))
        html = html.replace("{{IMPROVEMENTS}}", dept_item.get("待提升", "无"))
        html = html.replace("{{RECOMMENDATION}}", dept_item.get("任用建议", "无"))
        html = html.replace("{{WORK_CONTENT}}", dept_item.get("目前工作", "无"))
        html = html.replace("{{LIGHT}}", str(dept_item.get("红绿灯定位", "")))
        html = html.replace("{{POTENTIAL}}", dept_item.get("潜力等级", "无"))
        html = html.replace("{{SUCCESSOR}}", dept_item.get("继任者", "无"))
        
        # Light Color
        light = str(dept_item.get("红绿灯定位", ""))
        light_color = "#34a853" if "1" in light else "#fbbc04" if "2" in light else "#ea4335"
        html = html.replace("{{LIGHT_COLOR}}", light_color)
        
        # Data
        html = html.replace("{{INDICATORS}}", json.dumps(indicators, ensure_ascii=False))
        html = html.replace("{{SELF_DATA}}", json.dumps(self_values))
        html = html.replace("{{DEPT_DATA}}", json.dumps(dept_values))
        
        # Dev Report
        dev_report = get_dev_report(name)
        if dev_report:
            dev_section = f"""
            <div class="card full-width" style="background: #e8f0fe; border: 1px solid #1a73e8;">
                <h3 style="color: #1a73e8;">人才发展报告 (详述)</h3>
                <div style="white-space: pre-wrap; font-size: 14px; line-height: 1.6;">{dev_report}</div>
            </div>
            """
            html = html.replace("{{DEV_REPORT_SECTION}}", dev_section)
        else:
            html = html.replace("{{DEV_REPORT_SECTION}}", "")

        # Save
        file_name = f"2025年度评价报告-{name}.html"
        with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Generated {file_name}")

if __name__ == "__main__":
    generate_html()
