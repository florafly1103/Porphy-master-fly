from rxnfp.models import SmilesLanguageModelingModel, SmilesClassificationModel
from sklearn.model_selection import train_test_split

def Pre_training(args, train_file, val_file):
    #------初始化模型------
    # 使用无标签数据训练BERT模型，输入：大量SMILES字符串（无标签）
    model_pre_train = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args,use_cuda=False) 
    model_pre_train.train_model(train_file=train_file, eval_file=val_file)
    return model_pre_train
   
    

