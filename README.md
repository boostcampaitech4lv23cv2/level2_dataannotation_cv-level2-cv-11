# level2_dataannotation_cv-level2-cv-11
![image](https://user-images.githubusercontent.com/62556539/206107392-78a7a265-35d3-491a-99f9-01accbaf78d1.png)


## 프로젝트 개요

 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.
<img src="https://user-images.githubusercontent.com/62556539/210493623-61858a41-5910-4b17-8b69-5cc51037e673.png"  width="60%" height="60%"/>
(출처 : 위키피디아)

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.
- 예측 csv 파일 제출 (Evaluation) 방식이 아닌 **model checkpoint 와 inference.py 를 제출하여 채점**하는 방식입니다. (Inference) 상세 제출 방법은 AI Stages 가이드 문서를 참고해 주세요!
- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format

## 프로젝트 팀 구성

| [류건](https://github.com/jerry-ryu) | [심건희](https://github.com/jane79) | [윤태준](https://github.com/ta1231) | [이강희](https://github.com/ganghe74) | [이예라](https://github.com/Yera10) |
| :-: | :-: | :-: | :-: | :-: | 
| <img src="https://avatars.githubusercontent.com/u/62556539?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/48004826?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/54363784?v=4"  width="200"> | <img src="https://avatars.githubusercontent.com/u/30896956?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/57178359?v=4" width="200"> |  
|[Blog](https://kkwong-guin.tistory.com/)  |[Blog](https://velog.io/@goodheart50)|[Blog](https://velog.io/@ta1231)| [Blog](https://dddd.ac/blog) | [Blog](https://yedoong.tistory.com/) |

<div align="center">

![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)

</div align="center">

## 디렉토리 구조

```CMD
|-- README.md
|-- code
|   |-- convert_mlt.py
|   |-- dataset.py
|   |-- detect.py
|   |-- deteval.py
|   |-- east_dataset.py
|   |-- inference.py
|   |-- inference_viz.py
|   |-- loss.py
|   |-- model.py
|   |-- reproducibility.py
|   |-- requirements.txt
|   `-- train.py
`-- notebook
    |-- FiftyOne.ipynb
    |-- OCR_EDA.ipynb
    |-- assert.ipynb
    |-- convert_poly.ipynb
    |-- ufo_split_dataset.ipynb
    `-- utils
        `-- ufo.py
```


## 프로젝트 수행 환경
모든 실험은 아래의 환경에서 진행되었다.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-PCIE-32GB

## 타임라인

![Untitled (8)](https://user-images.githubusercontent.com/62556539/210493429-c9e82b27-850e-4915-b838-13584d8ba4b9.png)


## 프로젝트 수행 절차 및 방법

[![image](https://user-images.githubusercontent.com/62556539/200262300-3765b3e4-0050-4760-b008-f218d079a770.png)](https://excessive-help-ce8.notion.site/89fdd748c8f6465a9bb1775fa71040a8)

## 프로젝트 수행 결과

실험 종합 결과

dataset: ICDAR korea + boostcamp(검수)

image_size: 1024

crop: None

learning rate: 0.001

epoch:200

scheduler: steplr(100에서 0.0001로)

batch size: 8

Rotate: 10

optimizer: Adam
![image](https://user-images.githubusercontent.com/62556539/210494409-de4e05d6-53bd-455d-8159-bc3a65b019c2.png)

## 자체 평가 의견

**잘한 점**
- Github Convention을 정하여 협업 결과 공유 및 정리에 큰 도움이 되었다.
- Github 이슈, PR 등의 기능을 적극적으로 사용해보았다.
- 힘든 일정에도 서로를 격려하고 팀 분위기를 긍정적으로 유지하였다.
- 함께 디버깅하여 빠른 문제 대응을 할 수 있었다.
- Wandb를 사용하여 실시간 모니터링 및 팀원들과의 결과 공유가 용이했다.
- 문제 의식이나 issue들을 팀끼리 잘 공유하여 역할 분배를 잘 해내었다.

**아쉬운 점:**
- 추가한 코드에 대해 코드 내에서 설명이 부족했다.
- 모델 및 기법들에 대해 이론적인 공부와 결과분석이 부족했다.
- 대회 종료까지 Bounding Box의 크기, 비율을 고려하지 않은 데이터셋 분할을 사용해서 모델 학습과 검증이 잘 안되었다.
- augmentation 에 대한 실험이 부족했다.
- 팀원간의 방향성 공유가 부족했다.
- data 근거에 기반해서 실험계획이 수립되지 않았다.
- 재현성 결여, 데이터셋 노이즈로 인해 실험 결과를 잘 신뢰할 수 없었다.
- 사소한 버그들 때문에 실험, 검수 등에 문제가 있었다.

**개선할 점:**
- Commit하면 팀원들과 공유하기
- PM를 매일 돌아가면서 하기
- 실험 전에 브랜치를 파고 실험 후에 pull request하기
- 기능 구현하면 다같이 코드리뷰하고 Merge하기
---
