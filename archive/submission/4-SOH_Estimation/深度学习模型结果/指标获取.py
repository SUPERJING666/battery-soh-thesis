import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics():
    # 定义基础路径
    base_path = r"G:\boshi\1_boyi\3_ziji\My_program\long\3\code\4-SOH_Estimation\深度学习模型结果"
    
    # 定义要读取的文件名及其对应的模型名称
    files_to_process = {
        "CC": "result_CC.csv",
        "CV": "result_CV.csv",
        "RE": "result_RE.csv"
    }
    
    # 用于存储所有结果的列表
    results = []
    
    # 遍历字典，依次处理每个文件
    for model_name, file_name in files_to_process.items():
        file_path = os.path.join(base_path, file_name)
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 提取真实值(true)和预测值(pred)
                y_true = df['true']
                y_pred = df['pred']
                
                # 计算各项指标
                mse = mean_squared_error(y_true, y_pred) * 100
                rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
                mae = mean_absolute_error(y_true, y_pred) * 100
                r2 = r2_score(y_true, y_pred)
                
                # 将结果存入字典并添加到列表中
                results.append({
                    "model": model_name,
                    "RMSE": rmse,
                    "MSE": mse,
                    "MAE": mae,
                    "R2": r2
                })
                print(f"成功处理: {file_name}")
                
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")
        else:
            print(f"警告: 找不到文件 -> {file_path}")
            
    # 将结果列表转换为 DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # 定义输出文件的路径
        output_file = os.path.join(base_path, "all_models_evaluation_results.csv")
        
        # 保存结果为新的CSV文件，不保留行索引
        results_df.to_csv(output_file, index=False)
        print("\n计算完成！")
        print(f"所有模型的评价结果已保存至: {output_file}")
        
        # 在控制台打印预览结果
        print("\n结果预览:")
        print(results_df.to_string(index=False))
    else:
        print("\n未找到任何有效数据，未生成结果表格。")

if __name__ == "__main__":
    calculate_metrics()