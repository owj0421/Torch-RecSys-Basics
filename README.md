# <div align="center"> Torch Recsys Basics </div>

<div align="center"> Implementation of Basic Recommendation System Models with PyTorch </div>

## 🤗 Introduction

안녕하세요! 이곳은 기본부터 심화까지 추천시스템 모델을 구현하는 공간입니다. Feature Engineering과 같은 정확도를 높히기 위한 튜닝은 배제하고 모델의 정확한 구현에 중점을 두고 구현하였습니다. 모든 구현은 PyTorch로 되어있으며 Movielens 데이터셋을 사용해 평가했습니다. 추천시스템을 찾거나 공부하는 학생들에게 도움이 되길 바랍니다. 자유롭게 쓸 수 있으나, 만약 코드에 잘못된 부분이 있다면 꼭 알려주세요.<br>

### Datasets
[movielens-latest-small](https://grouplens.org/datasets/movielens/)<br>

## 📚 Implement Details
### Collaborative Filtering(Memory Based)
|Index|Model(Review)|RMSE|nDCG@10|HR@10|F1@10|
|:-:|:-|-:|-:|-:|-:|
|1    |[User-based CF]()|0|0|0|0|
|2    |[Item-based CF]()|0|0|0|0|

### Collaborative Filtering(Model Based)
|Index|Model(Review)|RMSE|nDCG@10|HR|F1@10|
|:-:|:-|-:|-:|-:|-:|
|1    |[SVD]()|0|0|0|0|
|2    |[Matrix Factorization]()|0|0|0|0|
|3    |[Neural Collaborative Filtering]()|0|0|0|0|

### Collaborative Filtering(AutoEncoder Based)
|Index|Model(Review)|RMSE|nDCG@10|HR|F1@10|
|:-:|:-|-:|-:|-:|-:|
|5    |[AutoRec]()|0|0|0|0|
|6    |[CDAE]()|0|0|0|0|
|7    |[EASE]()|0|0|0|0|
|8    |[RecVAE]()|0|0|0|0|


### Session Based(Sequential)
|Index|Model(Review)|HR|nDCG|MRR|
|:-:|:-|:-:|:-:|:-:|
|1    |[GRU4Rec]()|0|0|0|0|0|0|
|2    |[BERT4Rec]()|0|0|0|0|0|0|
|3    |[SASRec]()|0|0|0|0|0|0|

### Factorization Machine
기본적인 코드의 구성은 deepCTR을 참고했습니다.<br>
CTR Prediction이 아닌, 4점이상을 1, 미만을 0으로 한 Classification에 대해 학습한 결과 입니다.
|Index|Model(Review)|RMSE|F1|AUC|LogLoss|
|:-:|:-|:-:|:-:|:-:|:-:|
|1    |[Factorization machines](https://superficial-freeze-172.notion.site/Factorization-machines-85debc8b650a40f39156be320ec46a47?pvs=4)|0.428|0.345|0.714||
|2    |[Field Aware Factorization Machine]()|0.|0.|0.||
|3    |[Wide & Deep]()|0.413|0.468|0.740||
|4    |[Deep FM](https://superficial-freeze-172.notion.site/DeepFM-a-factorization-machine-based-neural-network-for-CTR-prediction-5891d516dbad413fb0da3e834c10771c?pvs=4)|0.408|0.467|0.752||
|5    |[Adaptive Factorization Network]()|0.|0.|0.||

## 🔔 Note
각 구현에 대한 논문 리뷰는 [여기서](https://superficial-freeze-172.notion.site/e20c78a9926b47e49d0921d229f64d4f?v=e3f1f712b2cf4abb94c14730710721cf&pvs=4) 보실 수 있습니다.<br>


