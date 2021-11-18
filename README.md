# data-annotation-nlp-level3-nlp-05

## `make_data.py`

- 데이터 생성
- `utils.download_tagtog(id, pw)`를 실행하여 tagtog zip 파일을 다운로드한 후 아래와 같은 명령어를 수행한다.

```bash
python make_data.py --data_type train
```

- `data_type`: 생성할 데이터의 종류
  - `relation`: relation class dictionary pickle 생성
  - `annotation`: relation annotating을 위한 데이터 생성
  - `iaa`: iaa 계산을 위한 데이터 생성
  - `train`: klue-re task 모델 학습을 위한 데이터 생성
- `test_split_ratio`: 전체 데이터를 훈련 및 테스트 데이터로 분리할 테스트 데이터의 비율
- `eval_split_ratio`: 훈련 데이터를 훈련 및 평가 데이터로 분리할 평가 데이터의 비율
- `seed`: 랜덤 시드