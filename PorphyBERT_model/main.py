import os
import pandas as pd
from Pre_training import Pre_training
from Fine_tuning import Fine_tuning
from inference import Prediction_evaluation, draw_regression_plot, draw_histplot

pre_type = 'HOMO' # HOMO、LUMO、E_gap
def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #  CUDA 调试配置，通常用于解决 GPU 训练中的内存占用问题
    model_pre_train_path = r'D:\software\miniconda3\envs\rxnfp_env\Lib\site-packages\rxnfp\models\transformers\bert_pretrained' # 加载预训练模型 (网上下载的) 
    # model_pre_train_path = 'PorphyBERT_model/Result/Pre_training' # 加载预训练模型 (自己训练的Pre_training) ????加后缀
    model_Fine_tuning_path = r'D:\software\miniconda3\envs\rxnfp_env\Lib\site-packages\rxnfp\models\transformers\bert_ft' # 加载微调模型 (网上下载的)
    # model_Fine_tuning_path = 'PorphyBERT_model/Result/Fine_tuning' # 加载微调模型 (自己训练的Fine_tuning) ????加后缀    跑完发现有一堆，我咋知道哪一个是best
    vocab_path = r'D:\software\miniconda3\envs\rxnfp_env\Lib\site-packages\rxnfp\models\transformers\bert_ft\vocab.txt'

    # ------模型的配置参数------
    config = {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,

    }

    # ------训练参数------
    args = {
        'config': config,
        'vocab_path': vocab_path,
        'train_batch_size': 16,
        'manual_seed': 42,
        'fp16': False,
        "num_train_epochs": 50,
        'max_seq_length': 300,
        'evaluate_during_training': True,
        'overwrite_output_dir': True,
        # 'output_dir': 'out/LUMO-self',
        'output_dir': 'PorphyBERT_model/Result/Pre_training',
        'learning_rate': 1e-4,
    }
    
    # ------数据集加载------  
    # 预训练，无监督，大量SMILES字符串（无标签）
    Pre_train_file = 'Dataset/spilt/df_PBDD_train.csv'
    Pre_val_file = 'Dataset/spilt/df_PBDD_val.csv'
    # 微调，监督，有标签数据（SMILES + HOMO值）
    Fine_tuning_train = pd.read_csv('Dataset/spilt/df_MpPD_train.csv')
    Fine_tuning_val = pd.read_csv('Dataset/spilt/df_MpPD_val.csv')
    Fine_tuning_test = pd.read_csv('Dataset/spilt/df_MpPD_test.csv')
    # 可视化
    infere_result_path = 'Infere_result/test_prediction_HOMO.csv'
    return (args, model_pre_train_path, model_Fine_tuning_path, Pre_train_file, Pre_val_file, Fine_tuning_train, Fine_tuning_val,Fine_tuning_test, infere_result_path)

if __name__ == '__main__':
    (args,
     model_pre_train_path,
     model_Fine_tuning_path,
     Pre_train_file,
     Pre_val_file,
     Fine_tuning_train,
     Fine_tuning_val,
     Fine_tuning_test,
     infere_result_path) = main()
    Mode = 'Pre_training' # 模式选择
    if Mode == 'Pre_training':
        print("开始预训练...")
        Pre_training(args, Pre_train_file, Pre_val_file)
        print("预训练完成...")
    elif Mode == 'Fine_tuning':
        print("开始训练微调模型...")
        Fine_tuning(model_pre_train_path, Fine_tuning_train, Fine_tuning_val, pre_type)
        print("微调模型训练完成...")
    elif Mode == 'Pre_evaluation':
        print("开始推理...")
        Prediction_evaluation(model_Fine_tuning_path, Fine_tuning_test)
        print("推理完成...")
        # 可视化
        # y_test = pd.read_csv(infere_result_path)[pre_type].values
        # y_pred = pd.read_csv(infere_result_path)['pre'].values
        # draw_regression_plot(y_test, y_pred, pre_type, name='test')
        # print("散点图可视化完成...")
        # draw_histplot(y_test, y_pred, pre_type, name='test')
        # print("直方图可视化完成...")
    else:
        print("default...")
    
