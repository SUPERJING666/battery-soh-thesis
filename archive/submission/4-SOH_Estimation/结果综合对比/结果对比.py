import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter # 导入百分比格式化工具

# 设置中文字体，防止图表中文乱码 (根据你的系统可能需要调整，这里默认使用SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

def load_and_process_data():
    """读取并处理数据，合并为一个DataFrame"""
    # 1. 读取人工特征数据并添加 stage 列
    df_cc = pd.read_excel(r'G:\boshi\1_boyi\3_ziji\My_program\long\3\code\4-SOH_Estimation\人工特征模型结果\CC\CC_模型评价指标.xlsx')
    df_cc['stage'] = 'CC'
    df_cc.rename(columns={'RMSE': 'rmse', 'MSE': 'mse', 'MAE': 'mae', 'R2': 'r2'}, inplace=True)
    df_cv = pd.read_excel(r'G:\boshi\1_boyi\3_ziji\My_program\long\3\code\4-SOH_Estimation\人工特征模型结果\CV\CV_模型评价指标.xlsx')
    df_cv['stage'] = 'CV'
    df_cv.rename(columns={'RMSE': 'rmse', 'MSE': 'mse', 'MAE': 'mae', 'R2': 'r2'}, inplace=True)
    df_re = pd.read_excel(r'G:\boshi\1_boyi\3_ziji\My_program\long\3\code\4-SOH_Estimation\人工特征模型结果\RE\RE_模型评价指标.xlsx')
    df_re['stage'] = 'RE'
    df_re.rename(columns={'RMSE': 'rmse', 'MSE': 'mse', 'MAE': 'mae', 'R2': 'r2'}, inplace=True)
    
    # 2. 读取深度学习结果
    df_cnn = pd.read_csv(r'4-SOH_Estimation\深度学习模型结果\all_models_evaluation_results.csv')
    # 修改列名和内容
    df_cnn.rename(columns={'model': 'stage', 'RMSE': 'rmse', 'MSE': 'mse', 'MAE': 'mae', 'R2': 'r2'}, inplace=True)
    df_cnn['model'] = 'CNN'
    
    # 3. 合并所有数据
    df_all = pd.concat([df_cc, df_cv, df_re, df_cnn], ignore_index=True)
    
    # 统一列的顺序，方便查看
    df_all = df_all[['model', 'stage', 'rmse', 'mse', 'mae', 'r2']]
    
    # 创建保存目录
    os.makedirs(r'4-SOH_Estimation\结果综合对比', exist_ok=True)
    # 保存合并后的数据，方便核对
    df_all.to_csv(r'4-SOH_Estimation\结果综合对比\merged_all_models_results.csv', index=False)
    print("数据合并完成，已保存至 'merged_all_models_results.csv'")
    return df_all

def plot_bar_charts(df):
    """图1：分组柱状图，单独子图对比 MAE, RMSE, MSE，显示为百分比并突出最优模型"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 取消整体标题和子图标题
    
    metrics = ['mae', 'rmse', 'mse']
    stages = ['CC', 'CV', 'RE']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        # 绘制柱状图，固定 x 轴的顺序
        sns.barplot(data=df, x='stage', y=metric, hue='model', ax=ax, palette='Set2', order=stages)
        
        # 不添加标题
        ax.set_title('')
        ax.set_xlabel('Stage', fontsize=12)
        # Y轴标签添加百分比符号
        ax.set_ylabel(f'{metric.upper()} (%)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 将 Y 轴设置为百分比显示 (因为数据已经是百分比数值如2.34，所以xmax设为100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        
        # 计算当前指标的最大值，用于给文字留出空间和计算偏移量
        metric_max = df[metric].max()
        y_offset = metric_max * 0.03
        
        # 突出显示每个阶段最好的模型（误差最小的）
        for stage_idx, stage in enumerate(stages):
            stage_data = df[df['stage'] == stage]
            if stage_data.empty: continue
            min_val = stage_data[metric].min()
            
            # 遍历图表中的柱子
            for p in ax.patches:
                height = p.get_height()
                # 排除空柱子
                if pd.isna(height) or height == 0: 
                    continue
                
                bar_center = p.get_x() + p.get_width() / 2.
                
                # seaborn 绘制的分组柱状图，每一组的中心 x 坐标大约为 0, 1, 2
                # 通过 round(bar_center) 可以判断该柱子属于哪个 stage，再匹配高度是否等于最小值
                if abs(round(bar_center) - stage_idx) < 0.1 and abs(height - min_val) < 1e-6:
                    # 突出显示：红色边框加粗
                    p.set_edgecolor('red')
                    p.set_linewidth(2.5)
                    # 数据已经是百分比数值，直接显示 height 即可
                    ax.text(bar_center, height + y_offset, f'{height:.2f}%\n★', 
                            ha="center", va="bottom", color="red", fontweight='bold', fontsize=11)

        # 稍微拉高 y 轴上限，防止顶部的文字被截断
        ax.set_ylim(0, metric_max * 1.25)
        
        # 图例处理
        if i != 2:
            if ax.get_legend() is not None:
                ax.get_legend().remove() # 前两个子图移除图例
        else:
            # 最后一个子图保留图例，并将其移到图表外侧
            ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1.02, 1))
            
    plt.tight_layout()
    plt.savefig(r'4-SOH_Estimation\结果综合对比\Plot1_BarCharts.png', dpi=300, bbox_inches='tight')
    print("图1 (误差柱状图) 已保存为 Plot1_BarCharts.png")
    plt.show()

if __name__ == "__main__":
    # 确保当前工作目录下有那4个CSV文件
    try:
        df_merged = load_and_process_data()
        plot_bar_charts(df_merged)
        print("\n所有操作执行完毕！")
    except FileNotFoundError as e:
        print(f"错误: 找不到文件。请确保数据文件与本脚本在同一目录下。\n详细信息: {e}")
    except Exception as e:
        print(f"发生错误: {e}")