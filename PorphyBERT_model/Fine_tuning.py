from rxnfp.models import SmilesLanguageModelingModel, SmilesClassificationModel
from sklearn.model_selection import train_test_split

def Fine_tuning(model_pre_train_path, train_df, df_val, pre_type):
    # 使用有标签数据对预训练模型进行微调，输入：预训练模型 + 有标签数据（SMILES + HOMO值）
    model_Fine_tuning = SmilesClassificationModel('bert', model_pre_train_path, num_labels=1, args={'regression': True, 'num_train_epochs': 50}, use_cuda=False) # 创建微调模型
    model_Fine_tuning.train_model(train_df[['smiles', pre_type]], output_dir='PorphyBERT_model/Result/Fine_tuning', eval_df=df_val) # 训练微调模型
    return model_Fine_tuning