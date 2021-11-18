import os
import argparse
import shutil
import pickle
import pandas as pd
from collections import Counter
from utils import get_tagtog_df, label_to_num, get_eng_name
from sklearn.model_selection import train_test_split

file_path = './file'
save_path = '/opt/ml/data/dataset'

def make_relation_class_dict(args):
    relation_class_df = pd.read_csv(os.path.join(file_path, 'relation_class.csv'))
    
    dict_label_to_num = {en: id for en, id in zip(relation_class_df['eng_name'], relation_class_df['id'])}
    dict_num_to_label = {v:k for k, v in dict_label_to_num.items()}
    
    with open(os.path.join(file_path, 'dict_label_to_num.pkl'), 'wb') as f:
        pickle.dump(dict_label_to_num, f)

    with open(os.path.join(file_path, 'dict_num_to_label.pkl'), 'wb') as f:
        pickle.dump(dict_num_to_label, f) 

def make_iaa_data(args):
    raw_df = pd.read_excel(os.path.join(file_path, 'iaa_tourist_spot_raw.xlsx'), engine='openpyxl')
    iaa_df = raw_df.copy()    
    
    dict_label_to_num = label_to_num()
    
    for column in iaa_df.columns:
        iaa_df[column] = iaa_df[column].apply(lambda x: get_eng_name(x))
        iaa_df[column] = iaa_df[column].apply(lambda x: dict_label_to_num[x])
        
    iaa_df.to_excel(os.path.join(file_path, 'iaa_tourist_spot.xlsx'), index=False, encoding='utf-8')

def make_annot_data(args):
    annot_df = get_tagtog_df()
    annot_df.to_excel(os.path.join(file_path, 'relation_annotation.xlsx'), index=False, encoding='utf-8')

def get_labels(df):
    labels = []
    for i, row in df.iterrows():
        label = None
        if row['correct'] == True:
            label = get_eng_name(row['relation_1'])
        else:
            counter = Counter([row[f"relation_{j}"] for j in range(1,4)])
            if max(counter.values()) != 2:
                label = 'none'
            else:
                label = get_eng_name([k for k, v in counter.items() if v == 2][0])
        labels.append(label)
    return labels

def post_process_df(df, flag):
    df = df.reset_index(drop=True)
    df['id'] = df.index
    if flag == 'test':
        df['label'] = [100 for _ in range(len(df))]    
    df = df[['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source']]

    return df

def make_train_data(args):
    tagtog_df = get_tagtog_df()
    relation_df = pd.read_excel('./file/relation_for_train_raw.xlsx', engine='openpyxl')
    
    merge_df = pd.merge(tagtog_df, relation_df, on=['id'])
    merge_df['exclude'] = merge_df['exclude'].fillna('유지')

    df = merge_df.copy() # 1683
    df['label'] = get_labels(df)
    df = df.loc[df['exclude'] == '유지', :] # 1600
    df = df.loc[df['label'] != 'none', :] # 1594
    df['source'] = ['wikipedia' for _ in range(len(df))]
    df = df.drop(
        columns=[
            'title', 'sentence_with_entity',
            'subject_entity_word', 'subject_entity_start_idx', 'subject_entity_end_idx', 'subject_entity_type',
            'object_entity_word', 'object_entity_start_idx', 'object_entity_end_idx', 'object_entity_type',
            'id', 'relation_1', 'relation_2', 'relation_3', 'correct', 'exclude'
        ]
    )

    train_df, test_df = train_test_split(df, test_size=args.test_split_ratio, random_state=args.seed, stratify=df['label'].tolist())
    train_df, eval_df = train_test_split(train_df, test_size=args.eval_split_ratio, random_state=args.seed, stratify=train_df['label'].tolist())

    print(f"train: {len(train_df)}, eval: {len(eval_df)}, test: {len(test_df)}")

    data_dict = {'train': train_df, 'eval': eval_df, 'test': test_df}
    for key in data_dict.keys():
        data_dict[key] = post_process_df(data_dict[key], key)

    save_train_path = os.path.join(save_path, 'train')
    save_test_path = os.path.join(save_path, 'test')

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        
    os.makedirs(save_train_path)
    os.makedirs(save_test_path)

    for key in data_dict.keys():
        if key in ['train', 'eval']:
            data_dict[key].to_csv(os.path.join(save_train_path, f"{key}.csv"), index=False, encoding="utf-8")
        else:
            data_dict[key].to_csv(os.path.join(save_test_path, f"test_data.csv"), index=False, encoding="utf-8")

def get_args():
    parser = argparse.ArgumentParser(description="make data arguments")
    parser.add_argument("--data_type", required=True, help="generation data type", choices=['relation', 'annotation', 'iaa', 'train'])
    parser.add_argument("--test_split_ratio", required=False, help="train test split ratio", default=0.05)
    parser.add_argument("--eval_split_ratio", required=False, help="train eval split ratio", default=0.1)
    parser.add_argument("--seed", required=False, help="random seed", default=42)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.data_type == 'relation':
        make_relation_class_dict(args)
    elif args.data_type == 'annotation':
        make_annot_data(args)
    elif args.data_type == 'iaa':
        make_iaa_data(args)
    elif args.data_type == 'train':
        make_train_data(args)