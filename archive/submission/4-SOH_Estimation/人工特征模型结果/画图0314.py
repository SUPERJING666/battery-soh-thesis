import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# 输入输出路径
input_base_dir = r"G:\boshi\1_boyi\3_ziji\My_program\long\3\code\4-SOH_Estimation\models"
output_base_dir = r"G:\boshi\1_boyi\3_ziji\My_program\long\3\code\4-SOH_Estimation\人工特征模型结果"

features = ['CC', 'CV', 'RE']
models = ['AdaBoost', 'GBDT', 'NN', 'RF', 'SVR', 'XGBoost']

# 为6个子图定义编号 (a), (b), (c)...
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# 预设高级配色
color_real = '#333333'  # 质感深灰（代表完美的 y=x 真实基准线）
color_pred = '#E64B35'  # NPG红（代表模型实际预测散点）
box_palette = "Set3"    # 箱型图的学术柔和配色

# ================= 3. 核心绘图逻辑 =================
for feature in features:
    feature_input_dir = os.path.join(input_base_dir, feature)
    
    # 在输出目录下为每个特征建立单独的文件夹，保持结构清晰
    feature_output_dir = os.path.join(output_base_dir, feature)
    os.makedirs(feature_output_dir, exist_ok=True)
    
    all_errors_data = []  # 用于收集箱型图的数据
    metrics_data = []     # 用于收集各项评价指标的数据
    
    # ------------------ 绘制：真实值与预测值对比 (2x3子图) ------------------
    # 论文常用尺寸：宽约15-18cm，高约10-12cm
    # 修改：sharex=False 使每个子图单独设置x轴范围和刻度
    fig_line, axes_line = plt.subplots(nrows=2, ncols=3, figsize=(20*cm, 12*cm), sharex=False, sharey=True)
    axes_line = axes_line.flatten() # 展平方便循环遍历
    
    for idx, model in enumerate(models):
        csv_path = os.path.join(feature_input_dir, f"result_{model}.csv")
        ax = axes_line[idx]
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            y_true = df['real']
            y_pred = df['pred']
            
            # --- 计算各项评价指标 ---
            mse = mean_squared_error(y_true, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
            mae = mean_absolute_error(y_true, y_pred) * 100
            r2 = r2_score(y_true, y_pred)
            
            # 将指标存入列表
            metrics_data.append({
                'model': model,
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            })
            
            # 计算误差 (Pred - Real)
            error = y_pred - y_true
            
            # 将误差数据加入列表用于后续画箱型图
            for e in error:
                all_errors_data.append({'Model': model, 'Error': e})
            
            # 绘制真实值与预测值的散点对比图 (X=真实值, Y=预测值)
            ax.scatter(y_true, y_pred, color=color_pred, s=15, alpha=0.6, label='预测值', zorder=2)
            
            # 绘制 y=x 理想参考线 (对角线：当预测值等于真实值时点会落在这条线上)
            line_min = min(y_true.min(), y_pred.min())
            line_max = max(y_true.max(), y_pred.max())
            ax.plot([line_min, line_max], [line_min, line_max], color=color_real, linewidth=1.5, linestyle='--', label='真实基准线', zorder=1)
            
        else:
            ax.text(0.5, 0.5, f'数据缺失:\n{model}', ha='center', va='center', transform=ax.transAxes)
            
        # 子图美化
        ax.set_title(f"{subplot_labels[idx]} {model}", loc='center', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 修改：为每个子图单独设置 x 和 y 轴刻度，nbins=5 通常会生成 4~6 个刻度
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
        # 修改：为每个子图单独添加 x 轴标签
        ax.set_xlabel('真实值', fontsize=11)
        
        # 仅在最左侧的一列添加 y 轴标签
        if idx % 3 == 0:
            ax.set_ylabel('预测值', fontsize=11)

    # 统一提取图例放在最上方或最下方
    handles, labels = axes_line[0].get_legend_handles_labels()
    fig_line.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=11)
    
    plt.tight_layout()
    # 保存真实值与预测值散点对比图
    line_save_path = os.path.join(feature_output_dir, f"{feature}_真实值与预测值对比.png")
    fig_line.savefig(line_save_path, dpi=600, bbox_inches='tight')
    plt.close(fig_line)
    
    # ------------------ 保存：评价指标到 Excel ------------------
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        metrics_save_path = os.path.join(feature_output_dir, f"{feature}_模型评价指标.xlsx")
        df_metrics.to_excel(metrics_save_path, index=False)
    
    # ------------------ 绘制：所有模型估计误差箱型图 ------------------
    if all_errors_data:
        df_errors = pd.DataFrame(all_errors_data)
        
        plt.figure(figsize=(12*cm, 8*cm))
        
        # 绘制箱线图 (显示均值，设置线宽)
        sns.boxplot(x='Model', y='Error', data=df_errors, 
                    palette=box_palette, width=0.5, 
                    linewidth=1.2, showmeans=True, 
                    meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"})
        
        # 添加 y=0 的基准参考线
        plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=0)
        
        # 去除英文，纯中文标签
        # 修改：x轴标签改为“估计模型”，并去掉 plt.title
        plt.xlabel('估计模型', fontsize=12)
        plt.ylabel('估计误差', fontsize=12)
        
        # 优化坐标轴刻度及网格
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        
        plt.tight_layout()
        # 保存箱型图
        box_save_path = os.path.join(feature_output_dir, f"{feature}_模型估计误差箱型图.png")
        plt.savefig(box_save_path, dpi=600, bbox_inches='tight')
        plt.close()

print(f"✅ 所有图表及Excel指标表计算完毕，已严格按照论文标准保存至:\n{output_base_dir}")