import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

# 设置中文字体，防止图表中文乱码 (根据您的系统可能需要调整，这里默认使用SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

def load_and_process_data():
    """读取、筛选并合并两个文件的数据"""
    
    # ================= 1. 处理第一个文件 (单模型结果) =================
    file1_path = r'4-SOH_Estimation\结果综合对比\merged_all_models_results.csv'
    df1 = pd.read_csv(file1_path)
    
    # 统一 stage 格式，去除可能存在的空格并转大写
    df1['stage'] = df1['stage'].astype(str).str.strip().str.upper()
    
    # 分离出 CNN 模型的数据
    df_cnn = df1[df1['model'] == 'CNN'].copy()
    
    # 获取除 CNN 以外的其他模型
    df_others = df1[df1['model'] != 'CNN'].copy()
    
    # 在除 CNN 外的模型中，提取每个 stage 中 RMSE 最小的模型
    # groupby 之后使用 idxmin() 找到最小值的索引，然后用 loc 提取完整行
    best_others_idx = df_others.groupby('stage')['rmse'].idxmin()
    df_best_others = df_others.loc[best_others_idx]
    
    # 组合 File 1 的最终提取数据
    df_file1_final = pd.concat([df_best_others, df_cnn], ignore_index=True)
    
    
    # ================= 2. 处理第二个文件 (融合模型结果) =================
    file2_path = r'5-fusion\model_evaluation_results.xlsx'
    df2 = pd.read_excel(file2_path)
    
    # 生成 model 列：将 fusion_stage 内容首字母大写
    df2['model'] = df2['fusion_stage'].astype(str).str.strip().str.capitalize()
    
    # 统一 stage 格式
    df2['stage'] = df2['stage'].astype(str).str.strip().str.upper()
    
    # 统一列名以匹配 File 1
    df2 = df2.rename(columns={
        'RMSE(%)': 'rmse',
        'MSE(%)': 'mse',
        'MAE(%)': 'mae',
        'R2': 'r2'
    })
    
    # 仅保留需要的列
    df2 = df2[['model', 'stage', 'rmse', 'mse', 'mae', 'r2']]
    df_file1_final = df_file1_final[['model', 'stage', 'rmse', 'mse', 'mae', 'r2']]
    
    # ================= 3. 合并所有数据 =================
    df_final = pd.concat([df_file1_final, df2], ignore_index=True)
    df_final.to_excel(r'6-结果对比/merged_results.xlsx', index=False)
    print("数据处理与合并完成！合并后的数据预览：")
    print(df_final.head(10))
    return df_final


def plot_comparison_bars(df):
    """绘制分组柱状图，单独子图对比 MAE, RMSE, MSE，突出最佳模型"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['mae', 'rmse', 'mse']
    stages = ['CC', 'CV', 'RE']
    
    # 获取图表中要展示的唯一模型列表，以保证图例颜色一致
    models = df['model'].unique()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 绘制分组柱状图
        sns.barplot(data=df, x='stage', y=metric, hue='model', ax=ax, 
                    palette='Set2', order=stages, hue_order=models)
        
        # 不添加子图标题
        ax.set_title('')
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel(f'{metric.upper()} (%)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 将 Y 轴设置为百分比显示 (数据已经是 2.34 形式，所以 xmax=100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        
        # 获取当前指标在所有数据中的最大值，用于设置 y_offset 和 y_lim
        metric_max = df[metric].max()
        y_offset = metric_max * 0.03
        
        # ==== 核心逻辑：找出并突出显示每个 Stage 中误差最小的模型 ====
        for stage_idx, stage in enumerate(stages):
            stage_data = df[df['stage'] == stage]
            if stage_data.empty: 
                continue
            
            # 找到当前 Stage 的最小误差值
            min_val = stage_data[metric].min()
            
            # 遍历坐标系里的所有柱子 (Patches)
            for p in ax.patches:
                height = p.get_height()
                
                # 跳过空柱子/高度无效的柱子
                if pd.isna(height) or height == 0: 
                    continue
                
                bar_center = p.get_x() + p.get_width() / 2.
                
                # 判断该柱子是否属于当前 stage (中心点坐标约为 0, 1, 2)
                # 并且高度是否匹配我们找到的最小值 (由于浮点数精度，使用微小差值比较)
                if abs(round(bar_center) - stage_idx) < 0.1 and abs(height - min_val) < 1e-6:
                    # 1. 边框加粗标红
                    p.set_edgecolor('red')
                    p.set_linewidth(2.5)
                    
                    # 2. 在柱子顶端打上星号及数值标签
                    ax.text(bar_center, height + y_offset, f'{height:.2f}%\n★', 
                            ha="center", va="bottom", color="red", fontweight='bold', fontsize=11)

        # 增加 y 轴上限，防止带有星号的文字被图片顶部截断
        ax.set_ylim(0, metric_max * 1.25)
        
        # ==== 图例处理 ====
        if i != 2:
            if ax.get_legend() is not None:
                ax.get_legend().remove() # 前两个子图不需要图例
        else:
            # 第三个子图(MSE)保留图例，放置在图表外侧
            ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1.02, 1))
            
    plt.tight_layout()
    
    # 保存图片
    save_dir = r'6-结果对比'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'Fusion_Comparison_BarCharts.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"对比图已成功生成并保存至：{save_path}")
    plt.show()

if __name__ == "__main__":
    try:
        # 执行数据处理与画图
        df_merged = load_and_process_data()
        plot_comparison_bars(df_merged)
    except FileNotFoundError as e:
        print(f"找不到文件，请检查相对路径是否正确。\n报错信息: {e}")
    except Exception as e:
        print(f"发生错误: {e}")