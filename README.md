# <div align="center"> Torch Recsys Implementation </div>

<div align="center"> Implementation of Recommendation System Models with PyTorch </div>

## 🤗 Introduction

안녕하세요! 이곳은 기본부터 심화까지 추천시스템 모델을 구현하는 공간입니다. 모든 구현은 PyTorch로 되어있으며 Movielens 데이터셋을 사용해 구현했습니다. 추천시스템을 찾거나 공부하는 학생들에게 도움이 되길 바랍니다. 자유롭게 쓸 수 있으나, 만약 코드에 잘못된 부분이 있다면 꼭 알려주세요.<br>

Hello! This is a space that implements recommended system models from basic to advanced. All implementations are in Pytorch and have been implemented using the movielens dataset. I hope it will be helpful for students who find or study the recommendation system. You can write freely, but if there is something wrong with the code, please let me know.<br>

### Datasets
[movielens-latest-small](https://grouplens.org/datasets/movielens/)<br>

## 📚 Implemented Paper Lists
### Collaborative Filtering(Memory-Based)
|Index|Model(Review)|Recall@20|nDCG@20|
|:-:|:-|:-:|:-:|
|1    |[User-based CF]()|         |         |
|2    |[Item-based CF]()|         |         |

### Collaborative Filtering(Model-Based)
|Index|Model(Review)|Recall@20|nDCG@20|
|:-:|:-|:-:|:-:|
|1    |[SVD]() |         |         |
|2    |[Matrix Factorization]() |         |         |
|3    |[Matrix Factorization(BPR)]() |         |         |
|4    |[Neural Collaborative Filtering]() |         |         |
|5    |[AutoRec]() |         |         |
|6    |[CDAE]() |         |         |
|7    |[EASE]() |         |         |
|8    |[RecVAE]() |         |         |


### Session-Based(Sequential)
|Index|Model(Review)|HR|nDCG|MRR|
|:-:|:-|:-:|:-:|:-:|
|1    |[GRU4Rec]()|         |         |         |
|2    |[BERT4Rec]()|         |         |         |
|3    |[SASRec]()|         |         |         |

### CTR Prediction
|Index|Model(Review)|RMSE|F1|AUC|LogLoss|
|:-:|:-|:-:|:-:|:-:|:-:|
|1    |[Factorization machines](https://superficial-freeze-172.notion.site/Factorization-machines-85debc8b650a40f39156be320ec46a47?pvs=4)|         |         |         |         |
|2    |[Wide & Deep]()|         |         |         |         |
|3    |[Deep FM](https://superficial-freeze-172.notion.site/DeepFM-a-factorization-machine-based-neural-network-for-CTR-prediction-5891d516dbad413fb0da3e834c10771c?pvs=4)|         |         |         |         |

## 🔔 Note
각 구현에 대한 논문 리뷰는 [여기서](https://superficial-freeze-172.notion.site/e20c78a9926b47e49d0921d229f64d4f?v=e3f1f712b2cf4abb94c14730710721cf&pvs=4) 보실 수 있습니다.<br>
A paper review of each implementation can be found at [here](https://superficial-freeze-172.notion.site/e20c78a9926b47e49d0921d229f64d4f?v=e3f1f712b2cf4abb94c14730710721cf&pvs=4). (Only Available in Korean) <br>


