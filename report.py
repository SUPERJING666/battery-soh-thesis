"""Generate thesis replacement text and the response letter from accepted runs."""
from pathlib import Path
import re,json,shutil,time,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT,Dataset,STAGES,CFG,metrics

def table(columns,rows):
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'

def fmt(mean,sd=None):
    return f'{mean:.3f}' if sd is None or pd.isna(sd) else f'{mean:.3f} ± {sd:.3f}'

def main():
    summary=pd.read_csv(ROOT/'reports/summary.csv');runs=pd.read_csv(ROOT/'reports/all_runs.csv')
    actual=len(list((ROOT/'results').glob('*/result.json')))
    assert actual==735,actual
    main=summary[(summary.scenario=='holdout')&(summary.track=='fair')&(summary.input_mode=='joint')]
    main=main[main.model.isin(['RF','GBDT','NN','AdaBoost','SVR','XGBoost','CNN','Early','Middle','Late'])]
    assert len(main)==30 and (main.n_runs==5).all()
    main_table=table(['模型','阶段','RMSE／百分点（均值±SD）','MAE／百分点（均值±SD）','R²（均值±SD）'],[[r.model,r.stage,fmt(r.rmse_mean,r.rmse_sd),fmt(r.mae_mean,r.mae_sd),fmt(r.r2_mean,r.r2_sd)] for _,r in main.sort_values(['stage','rmse_mean']).iterrows()])
    winners={s:main[main.stage==s].sort_values('rmse_mean').iloc[0] for s in STAGES}
    winner_text='；'.join(f'{s}阶段{r.model}的平均RMSE为{r.rmse_mean:.3f}±{r.rmse_sd:.3f}个百分点' for s,r in winners.items())+'。'
    cv=runs[runs.scenario.str.startswith('cv')].groupby(['stage','model']).agg(folds=('scenario','nunique'),rmse_mean=('rmse_pp','mean'),rmse_sd=('rmse_pp','std'),mae_mean=('mae_pp','mean')).reset_index()
    cv.to_csv(ROOT/'reports/cv_fold_summary.csv',index=False)
    pooled=[]
    for stage in STAGES:
        for model in ['XGBoost','CNN','Early','Middle','Late']:
            files=[ROOT/'results'/f'cv{i}__fair__joint__{stage}__{model}__s2025'/'predictions.csv' for i in range(5)]
            p=pd.concat([pd.read_csv(f) for f in files],ignore_index=True);assert len(p)==22250 and not p.duplicated(['id','cycle']).any()
            pooled.append({'stage':stage,'model':model,**metrics(p.true,p.pred)})
    pd.DataFrame(pooled).to_csv(ROOT/'reports/cv_pooled_metrics.csv',index=False)
    cv_table=table(['阶段','模型','5折RMSE均值±折间SD／百分点','折外覆盖样本数'],[[r.stage,r.model,fmt(r.rmse_mean,r.rmse_sd),22250] for _,r in cv.iterrows()])
    domain=summary[summary.scenario.str.startswith(('temp','rate'))]
    domain_table=table(['任务','阶段','模型','RMSE均值±训练SD／百分点','MAE均值±训练SD／百分点'],[[r.scenario,r.stage,r.model,fmt(r.rmse_mean,r.rmse_sd),fmt(r.mae_mean,r.mae_sd)] for _,r in domain.iterrows()])
    ablation=summary[(summary.scenario=='holdout')&(summary.track=='fair')&((summary.input_mode!='joint')|summary.model.isin(['conditions','MLP']))]
    ablation_table=table(['阶段','模型','输入设置','RMSE均值±SD／百分点'],[[r.stage,r.model,r.input_mode,fmt(r.rmse_mean,r.rmse_sd)] for _,r in ablation.iterrows()])
    robustness=pd.read_csv(ROOT/'reports/robustness.csv')
    rob=robustness.groupby(['test_type','level','stage','model']).agg(rmse_mean=('rmse_pp','mean'),rmse_sd=('rmse_pp','std')).reset_index();rob.to_csv(ROOT/'reports/robustness_summary.csv',index=False)
    imp=pd.read_csv(ROOT/'reports/permutation_importance.csv');importance=imp.groupby(['stage','model','branch']).agg(delta_mean=('delta_rmse_pp','mean'),delta_sd=('delta_rmse_pp','std'),count=('delta_rmse_pp','size')).reset_index();importance.to_csv(ROOT/'reports/importance_summary.csv',index=False)
    imp_table=table(['阶段','模型','置换分支','RMSE增量均值±联合波动／百分点','次数'],[[r.stage,r.model,r.branch,fmt(r.delta_mean,r.delta_sd),r['count']] for _,r in importance.iterrows()])
    benches=[]
    for p in (ROOT/'results').glob('*/benchmark_*.json'):benches.append(json.loads(p.read_text(encoding='utf-8')))
    bench=pd.DataFrame(benches);bench.to_csv(ROOT/'reports/complexity.csv',index=False)
    assert len(bench)==42,len(bench) # 30 CPU model-stage pairs + 12 neural GPU pairs.
    cpu=bench[bench.device=='cpu']
    bench_table=table(['模型','阶段','参数或结构规模','文件KiB','CPU完整预测中位ms','进程峰值MiB','加载及推理RSS增量MiB'],[[r.model,r.stage,r.structure_summary,f'{r.serialized_bytes/1024:.1f}',f'{r.preprocessing_and_forward_median_ms:.3f}',f'{r.process_peak_rss_bytes/2**20:.1f}',f'{r.rss_increment_bytes/2**20:.1f}'] for _,r in cpu.iterrows()])
    literature=[];matched=[]
    for p in (ROOT/'literature_results').glob('*/result.json'):
        meta=json.loads(p.read_text(encoding='utf-8'));row={k:meta[k] for k in ['run_id','reference','stage','seed','scenario','status']};row.update(meta['metrics']);literature.append(row)
        if meta['reference']==15:
            test_ids=set(meta['test_ids'])
            for model in ['XGBoost','CNN','Early','Middle','Late']:
                f=ROOT/'results'/f'{meta["scenario"]}__fair__joint__{meta["stage"]}__{model}__s{meta["seed"]}'/'predictions.csv'
                d=pd.read_csv(f);d=d[d.id.isin(test_ids)];assert len(d)==meta['metrics']['n']
                matched.append({'reference':15,'scenario':meta['scenario'],'stage':meta['stage'],'seed':meta['seed'],'model':model,'target_information':'no target adaptation used',**metrics(d.true,d.pred)})
    lit=pd.DataFrame(literature);lit.to_csv(ROOT/'reports/literature_metrics.csv',index=False);pd.DataFrame(matched).to_csv(ROOT/'reports/ref15_matched_controls.csv',index=False)
    assert len(lit)==95,len(lit)
    ls=lit.groupby(['reference','scenario','stage']).agg(runs=('seed','size'),rmse_mean=('rmse_pp','mean'),rmse_sd=('rmse_pp','std'),mae_mean=('mae_pp','mean')).reset_index()
    ls.to_csv(ROOT/'reports/literature_summary.csv',index=False)
    lit_table=table(['参考框架（均为公开描述对照）','任务','阶段','种子数','RMSE均值±SD／百分点'],[[int(r.reference),r.scenario,r.stage,int(r.runs),fmt(r.rmse_mean,r.rmse_sd)] for _,r in ls.iterrows()])
    # Aggregate metrics are always generated from accepted row-level predictions.
    verification=[]
    for p in (ROOT/'results').glob('*/result.json'):
        m=json.loads(p.read_text(encoding='utf-8'));d=pd.read_csv(p.parent/'predictions.csv');q=metrics(d.true,d.pred)
        assert abs(q['rmse_pp']-m['metrics']['rmse_pp'])<1e-8
        groups=[set(x['ids']) for x in m['split'].values()];assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        assert set(d.id)==set(m['split']['test']['ids']) and not d.duplicated(['id','cycle']).any()
        verification.append({'run_id':m['run_id'],'prediction_rows':len(d),'verified':True})
    (ROOT/'reports/final_result_checks.json').write_text(json.dumps({'verified_core_runs':verification,'literature_runs':len(lit),'benchmark_pairs':len(bench),'archive_hashes_verified':all(hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()==r['sha256'] for r in json.loads((ROOT/'archive/source_manifest.json').read_text(encoding='utf-8')))},ensure_ascii=False,indent=2),encoding='utf-8')
    # Scientific plots, no neural model or pipeline changes during reporting.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=['#4C78A8','#F58518','#54A24B','#E45756','#B279A2']
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,stage in zip(axes,STAGES):
        x=main[(main.stage==stage)&main.model.isin(['XGBoost','CNN','Early','Middle','Late'])].set_index('model').reindex(['XGBoost','CNN','Early','Middle','Late'])
        ax.bar(np.arange(5),x.rmse_mean,yerr=x.rmse_sd,capsize=3,color=colors);ax.set_xticks(np.arange(5),x.index,rotation=35,ha='right');ax.set_title(stage);ax.set_ylabel('RMSE (percentage points)');ax.grid(axis='y',alpha=.2)
    savefig(fig,'01_corrected_holdout');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,stage in zip(axes,STAGES):
        for color,model in zip(colors,['XGBoost','CNN','Early','Middle','Late']):
            x=rob[(rob.stage==stage)&(rob.model==model)&(rob.test_type=='gaussian')].sort_values('level');ax.errorbar(x.level*100,x.rmse_mean,yerr=x.rmse_sd,color=color,marker='o',capsize=2,label=model)
        ax.set_title(stage);ax.set_xlabel('Noise SD / training signal SD (%)');ax.set_ylabel('RMSE (percentage points)');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);savefig(fig,'02_noise_robustness');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,stage in zip(axes,STAGES):
        x=cv[cv.stage==stage].set_index('model').reindex(['XGBoost','CNN','Early','Middle','Late']);ax.bar(np.arange(5),x.rmse_mean,yerr=x.rmse_sd,capsize=3,color=colors);ax.set_xticks(np.arange(5),x.index,rotation=35,ha='right');ax.set_title(stage);ax.set_ylabel('Fold RMSE (percentage points)')
    savefig(fig,'03_group_cross_validation');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,stage in zip(axes,STAGES):
        x=importance[importance.stage==stage];models=['Early','Middle','Late'];width=.35
        for j,branch in enumerate(['curve','handcrafted_with_conditions']):
            xx=x[x.branch==branch].set_index('model').reindex(models);ax.bar(np.arange(3)+(j-.5)*width,xx.delta_mean,width,yerr=xx.delta_sd,capsize=2,label='Curve' if j==0 else 'Handcrafted + conditions')
        ax.set_xticks(range(3),models);ax.set_title(stage);ax.set_ylabel('Permutation RMSE increase (pp)');ax.axhline(0,color='black',linewidth=.5)
    axes[-1].legend(fontsize=8);savefig(fig,'04_branch_importance');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    tasks=['temp35','temp45','rate025','rate05','rate1'];models_order=['XGBoost','CNN','Early','Middle','Late']
    for ax,stage in zip(axes,STAGES):
        x=domain[domain.stage==stage].pivot(index='model',columns='scenario',values='rmse_mean').reindex(index=models_order,columns=tasks)
        im=ax.imshow(x.to_numpy(),aspect='auto',cmap='YlOrRd');ax.set_xticks(range(5),tasks,rotation=40,ha='right');ax.set_yticks(range(5),models_order);ax.set_title(stage)
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{x.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,shrink=.8,label='RMSE (pp)')
    savefig(fig,'05_unseen_conditions');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,stage in zip(axes,STAGES):
        x=ablation[ablation.stage==stage].copy();names=x.model+' / '+x.input_mode
        ax.bar(np.arange(len(x)),x.rmse_mean,yerr=x.rmse_sd,capsize=2,color='#4C78A8');ax.set_xticks(range(len(x)),names,rotation=55,ha='right',fontsize=7);ax.set_title(stage);ax.set_ylabel('RMSE (pp)')
    savefig(fig,'06_input_ablation');plt.close(fig)
    report='''# 修正与补充实验报告

本报告依据实际完成的训练及逐样本预测自动生成。历史送审结果保留在归档，不与新结果混用。模型排名不预设。

## 实验完成情况

- 核心实验735次：固定留出150次，保留原输入可用性的单种子修正30次，5折分组75次，跨工况375次，输入消融105次。
- 公开描述文献框架对照95次：[55]的RE对照5次，[61]的三阶段堆叠15次，[15]的五项跨工况三阶段对照75次，均5种子。
- 主结果每种模型5次训练，种子2025—2029；5折验证采用训练种子2025，折间波动另报。固定配置的确定性学习器可能得到相同指标，SD=0不代表其真实不确定性为0。
- 训练内筛选和标准化；传统模型不并入验证集；共同工况为温度及物理单位充电倍率。批量512、最多200轮和训练均值输出偏置是公开记录的新训练设置，不称为原实验逐项复现。
- 跨工况采用V2规则：源训练方差为零的工况列在所有集合置零，避免未见取值激活没有训练过的随机输入权重；该规则只读取训练方差。首次375项跨工况及75项对应文献框架结果保留在V1归档，不进入本报告。固定留出及五折的两项工况方差非零，输入等价后保留原检查点，等价记录与代码快照一并保存。
- 最终主实验采用V3有效输入接入：原有效卷积与池化会丢弃尾部拼接静态量。新公平CNN在151点时序编码后接入两项工况，早期融合将静态向量前置后进行卷积；225项受影响任务已重跑。中期、晚期、传统模型及仅曲线CNN的图与输入未改变，验证后保留其检查点。历史输入模式及旧结果仍在归档，不将原静态量实际无效的实验作为新公平比较。
- 12个保存模型的固定权重NumPy—PyTorch前向对照通过，最大差异小于5×10⁻⁵；数据分组与测试标签不参与特征选择检查通过。
- 上述12模型对照验证的是历史结构的转换；V3接入改动另用梯度路径检查，确认CNN两项工况及Early全部静态量真实连接输出，并复现了旧Early静态量梯度恒零的问题。
- 735个核心检查点已在原设备、原推理批形状下重载并核对全部预测，允许的最大归一化SOH差异为1×10⁻⁶；CPU与GPU默认卷积精度不同，CPU数值差异另作诊断，不宣称跨设备逐位一致。
- 单位核对：保存CV数值约875／1,750／3,500，与3.5 Ah电池0.25／0.5／1 C对应mA尺度一致。CV物理扰动标准差按mA报告，CC／RE按V报告，不能把保存CV数值直接标为A。

## 1. 修正后的固定留出结果

'''+main_table+f'按五种子平均RMSE排序：{winner_text}该排序不自动适用于MAE、跨电池或跨工况任务。\n\n'+'''## 2. 电池分组交叉验证

各电池及其循环只进入一个外层测试折。下表SD为五折之间的波动，不与固定划分五种子的训练波动混称。完整折外预测覆盖全部22,250条样本，整体加权指标另见`cv_pooled_metrics.csv`。

'''+cv_table+'''## 3. 未见工况测试

temp35和temp45为固定0.5 C、25℃训练至35℃和45℃测试；rate025、rate05、rate1为固定25℃、相应倍率完全留出。不同任务测试样本不同，误差不能直接当成仅由迁移导致的性能增量。完整结果如下。

'''+domain_table+'''## 4. 输入消融

conditions为仅工况；MLP-hi为仅人工特征；MLP-joint为人工特征与工况；CNN-curve为仅151点序列；融合no_conditions为人工特征与曲线而无工况。它们与主结果共同分析输入贡献，不能将跨模型结构变化都解释为纯特征贡献。

'''+ablation_table+'''## 5. 分支贡献与噪声

分支重要性基于相同温度／倍率内的整分支置换，5个训练种子各20次。下表SD同时包含训练和置换波动，原始表保留seed与repeat，可分别分析；不作因果贡献百分比解释。

'''+imp_table+'''噪声在物理信号上加入，人工特征重新提取。不同模型共享阶段噪声数组，标准差为训练信号标准差的0、0.5%、1%、2%；实际V／mA标准差见`robustness.csv`。仅清洁训练、噪声测试，未把噪声测试集用于训练。另有5%、10%、20%受控缺点后线性插值，仅说明已保存151点表示的输入敏感性，不能恢复原始变长序列处理。

## 6. 计算规模与耗时

CPU测试为批量1、预热20次、测量200次，完整预测包括特征计算和变换，不包括信号采集及文件读取。每个模型阶段使用独立进程测量RSS，列出进程峰值和加载模型后的增量。GPU另外测量相同12个神经网络阶段，使用同步计时及分配器峰值显存；详见`complexity.csv`。这些结果不等同于BMS实机达标验证。

'''+bench_table+'''## 7. 文献框架对照

**下表均为依据公开描述构建的本地对照，不是原论文严格复现。** 完整差异见《文献对照实现范围.md》。[55]只进行RE对照；[61]使用训练内部按电池分组OOF，GPR限制为2,000条训练内抽样并采用固定核；[15]使用独立无标签目标适配电池做源电池选择，普通直接迁移模型不使用这项信息，信息可用性不完全相同。比较[15]时，普通模型预测已重新筛至相同最终测试电池，数据见`ref15_matched_controls.csv`。

'''+lit_table+'''## 8. 未解决资料与结论边界

原始时间戳、阶段标记及151点构造过程尚未取得。人工特征窗口确认是1—149号点，RE_Max在保存数据中均等于RE_1，但无法据此确定距断流的物理时间。文献全文与作者代码细节不足，三个框架对照的精度不用于声称优于原论文。未进行真实BMS处理器部署；计时是当前工作站结果。

所有新结论仅依据上述实际结果；不保留历史Early-RE训练组0.90%的测试排名，不将未见工况或抗噪优势写成所有模型、所有任务均成立。新结果仅生成替换文本和图表，论文Word尚未自动修改。
'''
    (ROOT/'reports/补充实验报告.md').write_text(report,encoding='utf-8')
    replacements='''# 论文替换文本与图表

以下文本基于已完成的修正与补充实验，供同步论文使用；不表示Word正文已经改动。所有数值统一来自结果CSV。

## 第三章：数据与输入说明

本文使用66个完整电池标识对应的22,250条循环样本，按固定种子2025划分训练、验证与测试电池，分别包含39、13、14个标识以及13,762、4,448、4,040条样本。同一电池标识的循环不跨集合使用。人工特征在第1—149号点构成的窗口内计算，每个训练子集独立完成相关性筛选及标准化，验证与测试数据沿用已确定的变换。正式比较为各模型共同提供温度与充电倍率，恒定放电倍率不再作为共同输入。原始采样时间戳未包含在处理数据中，故不将RE_1直接等同于断流瞬间。

原CNN与Early结构在尾部拼接静态量后，由于有效卷积及池化的长度取整，静态量未能实际传递至最终表示。修正后CNN将两项工况接入时序编码后的回归头；Early将人工特征与工况组成的静态向量前置，再接入原卷积主干。梯度路径测试确认静态输入真实参与新模型计算。两项修改作为新实验版本记录，不将其误写为送审原实现。

保存CV电流数值按mA尺度使用，对应0.25 C、0.5 C和1 C的初始电流约875 mA、1,750 mA和3,500 mA；CC和RE电压按V使用。相关物理扰动和图轴单位统一核对，若以A显示，应将CV数值除以1,000。此单位说明不改变已标准化的训练输入和SOH误差。

对于跨工况任务，若温度或充电倍率在源训练组内没有变化，则在训练、验证及目标测试中统一将该标准化列置零，避免未经训练的任意系数影响目标估计。该判断仅依据源训练方差，不使用目标标签，也不假定能够从恒定源条件中学得温度或倍率的影响规律。

## 第四章：训练与评价说明

晚期融合在一个网络内构建曲线与人工特征分支，在标量层拼接后完成回归，采用共同MSE损失端到端训练。两条分支的梯度均由最终损失回传，未设置独立基学习器监督、显式置信度或门控权重，因此不作相应机理断言。神经网络训练最多200轮，批量512，Adam学习率0.001，以验证损失早停并保留最优权重。固定划分使用五个种子重复训练，另采用五折电池分组验证考察折外表现。

## 结果讨论与第五章结论

'''+winner_text+'''不同模型在平均误差、训练波动及未见工况任务中的表现存在差异，应结合具体阶段与输入条件选择方法。上述比较基于统一工况输入及修正后的训练流程，与送审稿原有结果分开报告。原早期融合RE误用了训练样本评价，其0.90%不再作为独立测试结果。噪声与排列试验用于分析当前数据范围内的输入敏感性，工作站计时用于描述计算需求，不外推为已完成车载或储能BMS部署。

## 替换图表

1. `01_corrected_holdout`：修正后固定留出五种子平均RMSE及标准差，替换原综合比较图中的模型精度部分。
2. `02_noise_robustness`：不同测量噪声等级的RMSE及训练波动，作为新增鲁棒性图。
3. `03_group_cross_validation`：五折电池分组验证及折间波动，作为新增泛化验证图。
4. `04_branch_importance`：同工况内整分支置换引起的误差变化，作为新增贡献度分析图。
5. `05_unseen_conditions`：未见温度及充电倍率的测试误差热图。
6. `06_input_ablation`：仅工况、人工分支、时序分支及去工况融合的对照。

每图提供PNG与SVG，位于`figures`。正文主结果表由`summary.csv`筛选holdout/fair/joint生成；完整排版表见《补充实验报告.md》。原表4-4、各阶段表及摘要、结论应同步使用同一新版本。
'''
    prose=json.loads((ROOT/'configs/manuscript_texts.json').read_text(encoding='utf-8'))
    replacements=prose['replacement_document']
    (ROOT/'reports/论文替换文本与图表.md').write_text(replacements,encoding='utf-8')
    update_letter(report,winner_text,main_table,cv_table,ablation_table,bench_table,lit_table)
    print('Final report, figures, acceptance checks and response letter generated.',flush=True)

def savefig(fig,name):
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=240,bbox_inches='tight')

def update_letter(report,winner_text,main_table,cv_table,ablation_table,bench_table,lit_table):
    path=ROOT.parent/'output/论文评阅意见逐条修改说明_Response_Letter.md';old=path.read_text(encoding='utf-8');backup=path.with_name('论文评阅意见逐条修改说明_Response_Letter_补实验前备份.md')
    if not backup.exists():shutil.copy2(path,backup)
    source=backup.read_text(encoding='utf-8')
    sections=re.split(r'^### 意见',source,flags=re.M)[1:12];assert len(sections)==11
    extraction=[]
    for sec in sections:
        opinion=sec.split('**评阅意见**')[1].split('**回复**')[0].strip();original=sec.split('**原文**')[1].split('**现有证据**')[0].strip();loc=sec.split('**修改位置**')[1].split('**修改后的内容**')[0].strip();extraction.append((opinion,original,loc))
    reportlink=(ROOT/'reports/补充实验报告.md').as_posix();text=f'''# 论文评阅意见逐条回复与修改说明（修正与补充实验版）

尊敬的评阅专家：

感谢您对论文的认真审阅与指导。依据评阅意见，已整理与送审稿结果对应的代码，修正原数据处理和评价问题，并补充固定留出重复训练、电池分组交叉验证、未见工况、输入消融、分支置换、噪声及计算开销实验。所有新指标来自保存的逐样本预测，不与历史结果混用。具体回复、修改理由、可用于论文的修改后内容及原文如下。

**使用说明：**本次更新修改说明、代码、实验结果与论文替换图表，论文Word正文尚未自动改动。原文和页码仍对应送审PDF，提交前需将修改文本同步至论文并刷新页码。核心实验735次、公开描述文献框架对照95次；后者因全文实现细节不足，明确不称为原论文严格复现。5个种子和5折分组验证分别报告训练波动与折间波动。

**总体验证与限制：**所有修正训练在电池标识分组后，仅在训练数据内筛选特征并拟合变换；Early-RE使用真正的测试组。旧版0.90%为训练组评价，永久撤出历史测试排名。原始时间戳仍缺失，不能确认首个RE样本距断流的时刻；工作站计时不等于BMS实机验证。完整逐样本结果、运行配置、图表及可复核总表见[补充实验报告]({reportlink})。

## 一、评阅专家一意见回复

对应评阅书：2122204506_2499409。

'''
    details=[
      ('统一指标并纠正Early-RE评价集合','感谢专家指出结论与指标展示问题。原Early-RE的0.90%经确认属于训练组，不能作为测试精度。已撤回该排名，使用正确集合重新训练，并以五种子指标统一模型比较。','评价样本与预处理改变后，应使用新版本结果整体替换，不能仅修改单个数值或把训练误差与测试误差共同排序。',f'修正后的公平比较结果如下。{winner_text}各模型的完整RMSE、MAE及R²见附录A。MSE以小数SOH尺度计算，RMSE、MAE列示为其小数尺度的100倍，单位为百分点；R²无量纲。原表“百分数MSE”不再沿用。'),
      ('明确划分并补充防泄漏及分组验证','感谢专家的建议。已保留原电池标识固定留出清单，并将特征筛选与Scaler拟合移至训练内部；人工特征模型不再合并验证数据训练。另完成五折电池分组验证，测试电池不参与早停、特征筛选或标准化。','原分组留出不能自动排除全数据特征筛选和CNN变换导致的问题，需要同时修正处理流程并验证外层电池隔离。','原数据按39／13／14个完整电池标识划分，对应13,762／4,448／4,040条样本；本次各模型共用这些集合。另将66个标识分成五个外层测试折，每个电池及其循环仅作为一个折的测试数据，开发组内部再按电池划出验证集。五折折外预测覆盖22,250条有效循环，结果见附录B。'),
      ('补充未见工况与公开描述域迁移对照','感谢专家的建议。已完成0.5 C条件下25℃至35℃、45℃的独立温度测试，以及25℃条件下三种充电倍率轮流留出。普通模型不接触目标工况训练数据。另依据文献[15]公开描述构建Wasserstein源电池选择与TCN-BiLSTM-Attention对照，明确目标无标签适配信息及实现差异。','已见混合工况留出与未见工况迁移不同；文献对照需要公开其信息使用条件，不能把本地适配实现写成原文严格复现。','不同温度及倍率任务独立划分电池，按源工况训练组筛选、标准化与模型拟合，目标工况只作普通模型最终测试。文献[15]框架对照另划出无标签目标适配电池，用于分布距离计算，最终测试电池独立；普通模型指标重新筛选至相同测试电池进行比较。由于原全文配置尚未完整取得，该对照仅支持公开描述框架下的本地比较，不宣称复现原SSTL方法。跨工况逐任务结果及信息边界见补充实验报告第3、7节。'),
      ('补充输入消融、分支重要性及复杂度','感谢专家的建议。已增加仅工况、仅人工特征MLP、人工特征加工况、仅时序CNN及去工况融合对照；采用新训练模型进行同工况内整分支置换。另测量参数量、序列化体积、CPU完整预测耗时与RSS，并单列GPU前向及分配器显存。','旧SHAP模型与主实验输入维度不一致，不能直接移植解释结果；消融和计算开销需使用新版本同一模型及数据。','消融结果见附录C；五个种子各进行20次同温度、同倍率内的整分支排列，不改变单条曲线内部顺序，分析误差增量而非因果贡献比例。CPU计时采用批大小1、20次预热及200次测量，并包含人工特征计算和输入变换；进程RSS与基线一同列示，GPU另测同步耗时及分配器峰值显存。相应数值见附录D，不外推为目标BMS已达实时部署要求。'),
      ('补充文献框架定量对照并限定贡献','感谢专家的建议。已完成文献[55]公开描述框架在RE阶段的5种子对照，以及文献[61]异构学习器与PCA补充特征框架的三阶段5种子对照。原文结构及训练细节不完整，因此明确列为公开描述本地对照，而非严格复现。','直接使用不同论文发表指标无法构成同协议比较；本地实现的资源近似和输入适配必须公开，贡献也不能由单一误差优势推断。','文献[55]对照采用并行多尺度卷积、注意力、统计特征和XGBoost；文献[61]对照采用训练内部按电池分组OOF，集成GPR、SVR、ELM和LSTM，并以PCA首主成分补充元学习器。GPR训练上限2,000条、阶段特征替代原完整特征等均列入差异说明。比较结果见附录E，仅用于本地明确条件下的框架比较。本文贡献表述为多运行阶段下不同融合层级的系统比较，不将某一结构或局部误差优势扩大为全面原创或普遍最优。'),
      ('补充完整数据处理协议','感谢专家指出预处理范围不清。已据代码整理完整电池清单，并在新实验中将筛选和标准化限定为训练数据；神经网络与传统模型共用固定训练、验证、测试归属。五折任务的每一外层训练组内部重新执行这些步骤。','原CNN在全数据归一化且测试时另拟合Scaler，人工特征筛选也使用全数据，不能靠补写规范说明消除影响，必须重新训练并分开报告历史结果。','每一数据划分先确定电池标识集合，再在训练数据内计算候选人工特征与容量的相关关系，选择三个特征并拟合各输入变换。验证数据仅用于训练早停，测试数据沿用训练确定的变换。源训练组内方差为零的工况列统一置零，不让目标未见取值激活未经训练的随机系数，该规则不使用目标标签。预处理器、特征名称、划分清单及模型和预测以同一运行编号保存。测试标签扰动不改变特征选择和Scaler的验收检查已通过。'),
      ('统一RE_Max的实际窗口定义','感谢专家指出定义矛盾。进一步核查确认统计特征在第1—149号点内计算，保存RE_Max在全部样本中等于RE_1。已据此修正窗口说明；原始时间戳未找到，故不能把RE_1解释成断流瞬间或断言其固定延迟。','151点阶段输入与149点统计窗口不同；列序号不能代替物理时间，也不能将有限采样窗口的最大值单独归因于欧姆内阻。','本文人工统计特征在阶段序列第1—149号点构成的有效窗口内计算。RE_Max为该窗口的最大电压，在现有处理数据中均对应RE_1；它不是原151点序列中必然的首点RE_0。由于原始时间戳缺失，尚不能确定RE_1距断流的物理时间。因此，将该量解释为特定处理窗口的上边界电压特征，不等同于断流瞬间电压，也不因充电截止电压固定便判定其无健康信息。相关性在各训练组内部重新评估，不保留旧全数据系数作为新筛选结果。'),
      ('明确输入、补充受控敏感性分析','感谢专家的建议。已明确原CNN为151点加两项工况的153维卷积输入，原主融合模型没有相同工况输入；新正式比较统一提供温度和充电倍率，并另列仅曲线及去工况对照。未取得原始变长曲线，故未冒称完成原始填充、截断或重采样复现。','在未知原始定长化规则时，不能把CNN阶段差异确定归因于填充或截断；公平性还需统一工况信息。','新实验保留151点阶段序列，对人工特征、CNN和融合结构共同提供温度及物理充电倍率；另通过无工况模型及人工分支MLP区分信息与学习器差异。测试端设置5%、10%、20%的受控采样点缺失，并用保留点线性插值恢复151点，只考察当前表示的输入敏感性。该实验不代替原始变长序列的填充或重采样验证，不以其推断未知的原始预处理机制。'),
      ('澄清晚期融合联合训练及梯度路径','感谢专家指出训练说明不足。已按代码和保存结构确认晚期融合为一个网络中的双分支标量层汇合，使用统一MSE损失端到端训练。新实验保留该结构，不改写为独立基学习器预训练或OOF元学习器。','结构上的分支分离不表示最终损失梯度隔离；标量节点没有独立监督，也没有显式置信度机制。','曲线分支与人工特征分支各经32节点及1节点全连接映射形成标量表示，拼接后经8节点和1节点层输出SOH。全部参数在共同MSE损失下联合更新，梯度经融合层回传两条分支。本文将其称为标量决策层融合，不再使用“避免梯度干扰”“动态置信度分配”或“独立预测后堆叠”的描述。文献[61]对照中的训练内部OOF属于另一模型，不能用于描述本文晚期融合。'),
      ('增加重复训练、噪声与未见工况验证','感谢专家的建议。已对固定留出和跨工况主要模型采用2025—2029五个种子，报告均值与样本标准差；另完成物理输入高斯噪声测试和五折电池分组验证。结论依据各任务实际误差和波动，不预设晚期融合抗噪或迁移性能最优。','原单次误差不足以支持抗噪和训练稳定性；折间变异与种子间变异也需分别统计。','噪声标准差设置为训练原信号标准差的0、0.5%、1%、2%，在测试物理量上加入，工况和SOH标签不变；所有模型共享同阶段扰动，人工统计量从扰动后曲线重新计算。清洁模型不在噪声测试数据上重新训练或拟合变换。固定划分报告五种子训练波动，五折验证单列折间差异，完整结果和图示见补充实验报告第2、3、5节。不将局部优势扩大为所有阶段和未见工况均鲁棒。'),
      ('统一全部新指标及模型排序','感谢专家指出跨表冲突。已完成历史120份预测核查，并从修正实验的逐样本预测统一生成新指标、图表和替换文本。送审历史误差不与新结果混用；Early-RE训练组误差从测试排名撤回，所有新模型排序依据五种子平均结果。','仅按出现次数修正数值不能解决不同版本和评价集合混用；需从同一运行记录及预测文件整体生成。',f'新主结果以附录A为准。{winner_text}各表使用小数尺度MSE、百分点RMSE和MAE及无量纲R²。负R²按定义保留，说明该任务的平方误差高于测试均值固定预测。摘要、第三章、第四章及第五章应共同引用新结果；不保留GBDT在CC最优、早期融合RE测试最优或晚期融合全面最优的历史表述。')
    ]
    graph_note='另经计算图与梯度核查，原CNN尾部工况量及Early尾部静态量会被卷积步长和池化丢弃。新公平CNN将工况量接入时序编码后的回归头，Early将静态向量前置后输入卷积，梯度检查确认有效接入；受影响225项任务已经重跑，旧版本单独归档。'
    for index in [3,7]:
        title,reply,reason,modified=details[index]
        details[index]=(title,reply+' '+graph_note,reason,modified+' '+graph_note)
    evidence_note=f'[完整实验报告与可复核结果]({reportlink})；每次核心运行保留代码哈希、环境、划分、变换器、训练日志、模型及逐样本预测。'
    prose=json.loads((ROOT/'configs/manuscript_texts.json').read_text(encoding='utf-8'))
    assert len(prose['body_blocks'])==11
    for i,(title,reply,reason,modified) in enumerate(details):
        modified=prose['body_blocks'][i].replace('\n\n','\n\n> ')
        if i==4:title='补充相关方法对照并突出研究贡献'
        if i==5:text+='## 二、评阅专家二意见回复\n\n对应评阅书：2122204506_2500964。\n\n'
        numeral=['一','二','三','四','五','一','二','三','四','五','六'][i];opinion,original,loc=extraction[i]
        text+=f'### 意见{numeral}：{title}\n\n**评阅意见**\n\n{opinion}\n\n**回复**\n\n{reply}\n\n**修改理由**\n\n{reason}\n\n**修改位置**\n\n{loc}\n\n**修改后的内容**\n\n> {modified}\n\n**原文**\n\n{original}\n\n**新增证据与边界**\n\n{evidence_note}\n\n'
    text+='## 三、新实验汇总附录\n\n### 附录A：公平输入固定留出五种子结果\n\n'+main_table+'### 附录B：五折电池分组验证\n\n'+cv_table+'### 附录C：输入消融\n\n'+ablation_table+'### 附录D：CPU计算规模及完整预测耗时\n\n'+bench_table+'### 附录E：公开描述文献框架对照\n\n以下均不是原论文严格复现，限制见完整报告及《文献对照实现范围.md》。\n\n'+lit_table
    text+='''## 四、仍需说明的限制

1. 原始时间戳与151点生成规则缺失，RE物理首点和原始变长预处理尚不能确定；已确认的统计窗口是1—149号点。
2. 文献[15]、[55]、[61]对照依据公开描述，存在明确输入、结构及资源近似，不能写成严格复现或对原论文的性能否定。
3. CPU／GPU耗时和内存为当前工作站条件，不等于目标BMS实机部署。
4. 固定划分五种子与五折验证分别衡量训练随机性和电池划分变化，不合并成虚构的25次独立试验。
5. 论文Word正文尚未自动修改，需根据已提供替换文本及图表同步，原文定位页码随后刷新。
'''
    assert text.count('**评阅意见**')==11 and text.count('**原文**')==11
    path.write_text(text,encoding='utf-8')

if __name__=='__main__':main()
