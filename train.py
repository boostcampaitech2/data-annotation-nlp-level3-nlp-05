import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer

from load_data import *
from utils import num_to_label, label_to_num
from inference import *


def klue_re_micro_f1(preds, labels):
  """KLUE-RE micro f1 (except no_relation)"""
  label_list = ['no_relation','loc:founded', 'loc:address', 'loc:heritage_info',
                'loc:alternative_name', 'loc:sites_contained', 'per:found', 'per:place_of_residence',
                'per:family', 'org:member']
  no_relation_label_idx = label_list.index("no_relation")
  label_indices = list(range(len(label_list)))
  label_indices.remove(no_relation_label_idx)
  return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
  """KLUE-RE AUPRC (with no_relation)"""
  labels = np.eye(10)[labels]

  score = np.zeros((10,))
  for c in range(10):
      targets_c = labels.take([c], axis=1).ravel()
      preds_c = probs.take([c], axis=1).ravel()
      precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
      score[c] = sklearn.metrics.auc(recall, precision)
  return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def train():
  # load model and tokenizer
  MODEL_NAME = "klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  TRAIN_DATASET_PATH = "/opt/ml/data/dataset/train/train.csv"
  EVAL_DATASET_PATH  = "/opt/ml/data/dataset/train/eval.csv"

  train_dataset = load_data(TRAIN_DATASET_PATH)
  eval_dataset  = load_data(EVAL_DATASET_PATH) if EVAL_DATASET_PATH is not None else train_dataset

  train_label = label_to_num(train_dataset['label'].values)
  eval_label  = label_to_num(eval_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_eval  = tokenized_dataset(eval_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_eval_dataset  = RE_Dataset(tokenized_eval, eval_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 10

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
  
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                  # model saving step.
    num_train_epochs=5,              # total number of training epochs
    learning_rate=5e-5,              # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,               # log saving step.
    evaluation_strategy='steps',     # evaluation strategy to adopt during training
    eval_steps = 500,                # evaluation step.
    load_best_model_at_end = True 
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_eval_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')

  ## load test datset
  test_dataset_dir = "/opt/ml/data/dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  create_csv(test_id, pred_answer, output_prob)
  print('---- Finish! ----')

def main():
  train()

if __name__ == '__main__':
  main()
