import os
import requests
import zipfile
import shutil
import json
import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

folder_path = './tagtog_result'

def download_tagtog(id, pw):
    # 로그인 위치
    url = 'https://tagtog.net/-login'

    # 다운로드 위치
    file_url = 'https://tagtog.net/nannullna/this-is-real/-downloads/dataset-as-anndoc'
    zip_file = 'download.zip'

    if os.path.exists(zip_file):
        os.remove(zip_file)
        
    # 로그인 정보
    login_info = {
        'loginid': id, # 아이디 입력
        'password': pw # 비밀번호 입력
    }

    # 로그인
    with requests.Session() as s:
        login_req = s.post(url, data=login_info)
        r = s.get(file_url)
        
        with open(zip_file, 'wb') as output:
            output.write(r.content)
            
    # 압축 파일 풀기
    zip_ = zipfile.ZipFile(zip_file)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    zip_.extractall(folder_path)

    os.remove(zip_file)    

def get_file_info(target_folder):
    target_folder = target_folder # 관광지

    # 폴더 경로
    root_path = os.path.join(folder_path, 'this-is-real')

    json_root_path = os.path.join(root_path, 'ann.json/master/pool')
    html_root_path = os.path.join(root_path, 'plain.html/pool')

    json_path = os.path.join(json_root_path, target_folder)
    html_path = os.path.join(html_root_path, target_folder)

    def get_unique_file_name(file_name):
        return file_name[:file_name[:file_name.rfind('.')].rfind('.')]

    # 파일명 목록
    file_list = [get_unique_file_name(file) for file in os.listdir(html_path)]

    # 파일 목록
    json_file_list = os.listdir(json_path)
    html_file_list = os.listdir(html_path)

    files = {}
    for file in file_list:
        files[file] = {'json': '', 'html': ''}

    for json_file in json_file_list:
        files[get_unique_file_name(json_file)]['json'] = os.path.join(json_path, json_file)
        
    for html_file in html_file_list:
        files[get_unique_file_name(html_file)]['html'] = os.path.join(html_path, html_file)

    # annotation_legend
    annotation_legend = os.path.join(root_path, 'annotations-legend.json')
    with open(annotation_legend, 'r') as f:
        annotation_legend = json.load(f)
    
    return files, annotation_legend  

def get_tagtog_df():
    files, annotation_legend = get_file_info('관광지')
    
    data = {
        'title': [],
        'sentence': [],
        'sentence_with_entity': [],
        'subject_entity': [],
        'object_entity': [],
        'subject_entity_word': [],
        'subject_entity_start_idx': [],
        'subject_entity_end_idx': [],
        'subject_entity_type': [],
        'object_entity_word': [],
        'object_entity_start_idx': [],
        'object_entity_end_idx': [],
        'object_entity_type': [],
        'confirm': []
    }

    for i, key in enumerate(files.keys()):
        # get title and sentence information from html file
        with open(files[key]['html'], 'r') as f:
            html_obj = f.read()
            
        bs_obj = BeautifulSoup(html_obj, 'html.parser')
        title, sentence = [obj.text for obj in bs_obj.select('pre')]

        data['title'].append(title)
        data['sentence'].append(sentence)


        # get entity information from json file
        entities = {
            'subj': {'word': None, 'start_idx': -1, 'end_idx': -1, 'type': None},
            'obj': {'word': None, 'start_idx': -1, 'end_idx': -1, 'type': None}
        }

        if files[key]['json'] != '':
            with open(files[key]['json'], 'r') as f:
                json_obj = json.load(f)
                
            for entity in json_obj['entities']:
                e_info, e_type = annotation_legend[entity['classId']].split('_')
                entities[e_info]['word'] = entity['offsets'][0]['text']
                entities[e_info]['start_idx'] = entity['offsets'][0]['start']
                entities[e_info]['end_idx'] = entity['offsets'][0]['start'] + len(entity['offsets'][0]['text']) - 1
                entities[e_info]['type'] = e_type

        data['subject_entity'].append(entities['subj'] if entities['subj']['word'] is not None else None)
        data['subject_entity_word'].append(entities['subj']['word'])
        data['subject_entity_start_idx'].append(entities['subj']['start_idx'])
        data['subject_entity_end_idx'].append(entities['subj']['end_idx'])
        data['subject_entity_type'].append(entities['subj']['type'])
        data['object_entity'].append(entities['obj'] if entities['obj']['word'] is not None else None)
        data['object_entity_word'].append(entities['obj']['word'])
        data['object_entity_start_idx'].append(entities['obj']['start_idx'])
        data['object_entity_end_idx'].append(entities['obj']['end_idx'])
        data['object_entity_type'].append(entities['obj']['type']) 

        # get sentence with entities information
        sentence_w_entity = sentence
        entities['subj']['symbol'] = '$$'
        entities['obj']['symbol'] = '@@'
        
        entity_list = sorted([val for val in entities.values()], key=lambda x: x['start_idx'], reverse=True)
        for entity in entity_list:
            if entity['word'] != '':
                b_str = sentence_w_entity[:entity['start_idx']]
                e_str = sentence_w_entity[entity['start_idx']:entity['end_idx']+1]
                a_str = sentence_w_entity[entity['end_idx']+1:]            
                sentence_w_entity = b_str + entity['symbol'] + e_str + entity['symbol'] + a_str    
        data['sentence_with_entity'].append(sentence_w_entity)
        
        # confirm
        data['confirm'].append(json_obj['anncomplete'])

    df = pd.DataFrame(data)
    df = df.sort_values('title')
    df = df.loc[df['subject_entity'].apply(lambda x: type(x)) == dict, :]
    df = df.loc[df['object_entity'].apply(lambda x: type(x)) == dict, :]
    df = df.loc[df['confirm'] == True, :]
    df = df.reset_index(drop=True)        
    df["id"] = [idx+1 for idx in df.index]
    df = df.drop(columns=['confirm'])
    
    return df

def get_label_to_num():
    with open('./file/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    return dict_label_to_num

def get_num_to_label():
    with open('./file/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    return dict_num_to_label