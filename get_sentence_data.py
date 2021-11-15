import os
import kss
from tqdm import tqdm
import pandas as pd

def main(file_path, save_file_name):
    """ner tagging이 포함되지 않은 데이터 생성"""
    
    file_list = os.listdir(file_path)
    
    tourist_spot_data = dict()

    for file_name in tqdm(file_list, total=len(file_list)):
        with open(os.path.join(file_path, file_name), 'r', encoding='utf-8') as f:
            data = f.read()
            sentences = []
            for sent in kss.split_sentences(data):
                sentences.append(sent)
            tourist_spot_data[file_name.split('.')[0]] = sentences
            
    data = {
        'title': [],
        'sentence': [],
        'status': []
    }

    for title in tqdm(tourist_spot_data.keys(), total=len(tourist_spot_data)):
        for sent in tourist_spot_data[title]:
            data['title'].append(title)
            data['sentence'].append(sent)
            data['status'].append(True if '.' in sent else False)

    df = pd.DataFrame(data)
    
    df.to_csv(save_file_name + '.csv', index=False, encoding='utf-8')
    df.to_excel(save_file_name + '.xlsx', index=False, encoding='utf-8')

if __name__ == '__main__':
    file_path = '../data/tourist_spot/preprocessed' # txt 파일들이 모여 있는 폴더 경로를 입력
    save_file_name = 'tourist_spot_preprocessed' # 저장될 excel 파일명을 입력
    
    main(file_path, save_file_name)