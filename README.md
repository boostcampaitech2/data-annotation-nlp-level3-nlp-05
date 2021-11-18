# data-annotation-nlp-level3-nlp-05

## `make_data.py`

- 데이터 생성

```bash
python make_data.py --data_type [annotation/iaa/train]
```

- `data_type` : 생성할 데이터의 종류
  - `annotation`: relation annotating을 위한 데이터 생성
  - `iaa`: iaa 계산을 위한 데이터 생성
  - `train`: klue-re task 모델 학습을 위한 데이터 생성
