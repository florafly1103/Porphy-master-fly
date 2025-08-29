import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rxnfp.models import SmilesLanguageModelingModel, SmilesClassificationModel

# ------预测并保存结果------
def predict_data(model_name, data, name):
    predict_list = model_name.predict(data['smiles'].to_list())[0]
    datas = data.copy()
    datas.insert(2, 'pre', predict_list)
    datas.to_csv(name, index_label=False, index=False)
    return 0

def Prediction_evaluation(model_Fine_tuning_path, df_test):  
    # 加载微调模型
    model_Fine_tuning = SmilesClassificationModel('bert', model_Fine_tuning_path, num_labels=1, args={'regression': True}, use_cuda=False)  
    # 预测
    print("开始预测...")
    predict_data(model_Fine_tuning, df_test, 'infere_result/MpPD-test_infere.csv')
    print("预测完成!")

# ------结果可视化------
# 散点图
def draw_regression_plot(y_test, y_pred, pre_type, name='test'):
    sns.set_style('darkgrid')
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(16, 16), dpi=300)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    # 指标标注
    r2_patch = mpatches.Patch(label="$\mathregular{R^2}$" + " = {:.4f}".format(r2), color='#0F59A4')
    rmse_patch = mpatches.Patch(label="RMSE = {:.4f} eV".format(rmse), color='#1661AB')
    mae_patch = mpatches.Patch(label="MAE = {:.4f} eV".format(mae), color='#3170A7')
    # 设置坐标轴范围
    if pre_type == 'E_gap':
        plt.xlim(1.6, 3.4)
        plt.ylim(1.6, 3.4)
    if pre_type == 'LUMO':
        plt.xlim(-3.9, -1.6)
        plt.ylim(-3.9, -1.6)
    if pre_type == 'HOMO':
        plt.xlim(-6.9, -4.1)
        plt.ylim(-6.9, -4.1)

    plt.tick_params(labelsize=40)
    # 绘制散点图范围 
    sns.scatterplot(x=y_pred, y=y_test, alpha=0.8, color='#144A74', linewidth=0, s=400)
    # 绘制对角虚线
    x_line = np.linspace(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()), 500)
    sns.lineplot(x=x_line, y=x_line, ls="--", color='#0F1423', alpha=0.3)
    # 添加图例和标签
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=30)
    ax.set_ylabel('Measured ' + pre_type + ' (eV)', fontsize=35, labelpad=30)
    ax.set_xlabel('Predicted ' + pre_type + ' (eV)', fontsize=40, labelpad=30)
    ax.set_title(name, fontsize=50, pad=30)
    # 保存图像并清理图形
    fig.savefig('Infere_result/' + name + '_regression.png')
    plt.cla()
    return

# 分布直方图
def draw_histplot(y_test, y_pred, pre_type,  name='test'):
    plt.figure(figsize=(16, 16), dpi=300)
    sns.color_palette("Paired", 8)
    sns.distplot(y_test, color='#20763B', label='Measured_'+ pre_type, norm_hist=False, kde=False,
                 hist_kws=dict(edgecolor='#20763B', linewidth=0),
                 bins=13)
    sns.distplot(y_pred, color='#081EAD', label='Predicted_'+ pre_type, norm_hist=False, kde=False,
                 hist_kws=dict(edgecolor='#081EAD', linewidth=0),
                 bins=13)
    plt.legend(fontsize=40)
    plt.title(name, fontsize=50, pad=30)
    plt.tick_params(labelsize=38)
    plt.xlabel(pre_type + "(eV)", fontsize=40, labelpad=30)
    plt.ylabel("Amount", fontsize=50, labelpad=30)
    plt.savefig('Infere_result/' + name + '_distplot.png', dpi=300)
    plt.cla()
    return
