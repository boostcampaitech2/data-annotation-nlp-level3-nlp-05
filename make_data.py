import os
import argparse
import re
import pickle
import pandas as pd
from utils import get_tagtog_df, get_label_to_num, get_num_to_label

file_path = './file'

def get_eng_name(data):
    result = re.findall(r"\([a-z]+:[a-z_]+\)|\([a-z_]+\)", data)[0]
    result = result.replace('(','').replace(')','')
    return result

def make_relation_class_dict():
    relation_class_df = pd.read_csv(os.path.join(file_path, 'relation_class.csv'))
    
    dict_label_to_num = {en: id for en, id in zip(relation_class_df['eng_name'], relation_class_df['id'])}
    dict_num_to_label = {v:k for k, v in dict_label_to_num.items()}
    
    with open(os.path.join(file_path, 'dict_label_to_num.pkl'), 'wb') as f:
        pickle.dump(dict_label_to_num, f)

    with open(os.path.join(file_path, 'dict_num_to_label.pkl'), 'wb') as f:
        pickle.dump(dict_num_to_label, f) 

def make_iaa_data():
    raw_df = pd.read_excel(os.path.join(file_path, 'iaa_tourist_spot_raw.xlsx'), engine='openpyxl')
    iaa_df = raw_df.copy()    
    
    dict_label_to_num = get_label_to_num()
    
    for column in iaa_df.columns:
        iaa_df[column] = iaa_df[column].apply(lambda x: get_eng_name(x))
        iaa_df[column] = iaa_df[column].apply(lambda x: dict_label_to_num[x])
        
    iaa_df.to_excel(os.path.join(file_path, 'iaa_tourist_spot.xlsx'), index=False, encoding='utf-8')

def make_annot_data():
    annot_df = get_tagtog_df()
    annot_df.to_excel(os.path.join(file_path, 'relation_annotation.xlsx'), index=False, encoding='utf-8')

def make_train_data():
    print('train')
    # df = get_tagtog_df()
    # print(df.head())

def get_args():
    parser = argparse.ArgumentParser(description="make data arguments")
    parser.add_argument("--data_type", required=True, help="generation data type", choices=['relation', 'annotation', 'iaa', 'train'])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.data_type == 'relation':
        make_relation_class_dict()
    elif args.data_type == 'annotation':
        make_annot_data()
    elif args.data_type == 'iaa':
        make_iaa_data()
    elif args.data_type == 'train':
        make_train_data()