import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义基础路径，请确保运行脚本时相对路径正确，或替换为绝对路径
base_dir = r"5-fusion\rusult"

# 定义需要遍历的文件夹和文件后缀
fusion_stages = ['early', 'middle', 'late']
stages = ['CC', 'CV', 'RE']

# 用于存储所有结果的列表
results_data = []

# 遍历各个阶段的文件夹
for fusion_stage in fusion_stages:
    folder_path = os.path.join(base_dir, fusion_stage)
    
    # 遍历每个文件夹下的三种文件
    for stage in stages:
        file_name = f"result_{stage}.csv"
        file_path = os.path.join(folder_path, file_name)
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 提取真实值(SOH)和预测值(pred)
                y_true = df['SOH']
                y_pred = df['pred']
                
                # 计算各项评估指标 (根据您的要求，MSE, MAE, RMSE 乘以 100 转换为百分比)
                mse = mean_squared_error(y_true, y_pred) * 100
                mae = mean_absolute_error(y_true, y_pred) * 100
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
                
                # 将结果添加到字典中
                results_data.append({
                    'fusion_stage': fusion_stage,
                    'stage': stage,
                    'MSE(%)': mse,
                    'MAE(%)': mae,
                    'RMSE(%)': rmse,
                    'R2': r2
                })
            except Exception as e:
                print(f"读取或计算文件 {file_path} 时出错: {e}")
        else:
            print(f"警告：未找到文件 {file_path}")

# 将结果列表转换为 DataFrame
results_df = pd.DataFrame(results_data)

# 将 DataFrame 保存为 Excel 文件
output_file = '5-fusion/model_evaluation_results.xlsx'
results_df.to_excel(output_file, index=False)

print(f"处理完成！所有结果已成功保存至: {os.path.abspath(output_file)}")