import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# ================= 1. 全局论文风格配置 =================
# 设置字体：强制使用宋体 (SimSun) 以解决中文不显示(方块)的问题
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman'] # 增加保底配置
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12              # 全局字号设定
plt.rcParams['axes.linewidth'] = 1.0        # 坐标轴线宽
plt.rcParams['xtick.direction'] = 'in'      # 刻度线向内
plt.rcParams['ytick.direction'] = 'in'      # 刻度线向内

cm = 1 / 2.54  # 厘米转英寸的换算系数

# ================= 2. 路径与变量定义 =================
# 根据要求修改为深度学习模型结果的路径
# 注意：若运行报错找不到路径，请将 input_base_dir 修改为绝对路径
input_base_dir = r"4-SOH_Estimation\深度学习模型结果"
output_base_dir = r"4-SOH_Estimation\深度学习模型结果\绘图输出"
os.makedirs(output_base_dir, exist_ok=True)

features = ['CC', 'CV', 'RE']

# 为3个子图定义编号 (a), (b), (c)
subplot_labels = ['(a)', '(b)', '(c)']

# 预设高级配色 (与原脚本一致)
color_real = '#333333'  # 质感深灰（代表完美的 y=x 真实基准线）
color_pred = '#E64B35'  # NPG红（代表模型实际预测散点）
box_palette = "Set3"    # 箱型图的学术柔和配色

all_errors_data = []  # 用于收集箱型图的数据

# ================= 3. 绘制：真实值与预测值对比散点图 (1x3子图) =================
# 尺寸：宽20cm，高10cm，1行3列
fig_line, axes_line = plt.subplots(nrows=1, ncols=3, figsize=(20*cm, 10*cm), sharex=False, sharey=True)

# 遍历三个特征的结果文件
for idx, feature in enumerate(features):
    csv_path = os.path.join(input_base_dir, f"result_{feature}.csv")
    ax = axes_line[idx]
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # 计算误差 (Pred - True) 注意新表头的真实值为 'true'
        error = df['pred'] - df['true']
        
        # 将误差数据加入列表用于后续画箱型图
        for e in error:
            all_errors_data.append({'Feature': feature, 'Error': e})
        
        # 绘制真实值与预测值的散点对比图 (X=真实值, Y=预测值)
        ax.scatter(df['true'], df['pred'], color=color_pred, s=15, alpha=0.6, label='预测值', zorder=2)
        
        # 绘制 y=x 理想参考线
        line_min = min(df['true'].min(), df['pred'].min())
        line_max = max(df['true'].max(), df['pred'].max())
        ax.plot([line_min, line_max], [line_min, line_max], color=color_real, linewidth=1.5, linestyle='--', label='真实基准线', zorder=1)
        
    else:
        ax.text(0.5, 0.5, f'数据缺失:\n{feature}', ha='center', va='center', transform=ax.transAxes)
        
    # 子图美化
    ax.set_title(f"{subplot_labels[idx]} {feature}", loc='center', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 设置刻度
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    # 坐标轴标签
    ax.set_xlabel('真实值', fontsize=11)
    if idx == 0:
        ax.set_ylabel('预测值', fontsize=11)

# 统一提取图例放在最上方
handles, labels = axes_line[0].get_legend_handles_labels()
fig_line.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11)

plt.tight_layout()
# 保存真实值与预测值散点对比图
line_save_path = os.path.join(output_base_dir, "深度学习模型_真实值与预测值对比.png")
fig_line.savefig(line_save_path, dpi=600, bbox_inches='tight')
plt.close(fig_line)


# ================= 4. 绘制：所有特征估计误差箱型图 =================
if all_errors_data:
    df_errors = pd.DataFrame(all_errors_data)
    
    plt.figure(figsize=(10*cm, 8*cm))
    
    # 绘制箱线图
    sns.boxplot(x='Feature', y='Error', data=df_errors, 
                palette=box_palette, width=0.5, 
                linewidth=1.2, showmeans=True, 
                meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"})
    
    # 添加 y=0 的基准参考线
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    
    # 轴标签设置
    plt.xlabel('阶段', fontsize=12)
    plt.ylabel('估计误差', fontsize=12)
    
    # 优化坐标轴刻度及网格
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    plt.tight_layout()
    # 保存箱型图
    box_save_path = os.path.join(output_base_dir, "深度学习模型_特征误差箱型图.png")
    plt.savefig(box_save_path, dpi=600, bbox_inches='tight')
    plt.close()

print(f"✅ 所有图表渲染完毕，已严格按照论文标准保存至:\n{output_base_dir}")