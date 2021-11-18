import pandas as pd
import pickle

def main():
    relation_class_df = pd.read_csv('./data/relation_class.csv')
    
    dict_label_to_num = {en: id for en, id in zip(relation_class_df['eng_name'], relation_class_df['id'])}
    dict_num_to_label = {v:k for k, v in dict_label_to_num.items()}
    
    with open('./data/dict_label_to_num.pkl', 'wb') as f:
        pickle.dump(dict_label_to_num, f)

    with open('./data/dict_num_to_label.pkl', 'wb') as f:
        pickle.dump(dict_num_to_label, f)    

if __name__ == "__main__":
    main()