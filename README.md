# level2_objectdetection_cv-level2-cv-11
![image](https://user-images.githubusercontent.com/62556539/206107392-78a7a265-35d3-491a-99f9-01accbaf78d1.png)


## 프로젝트 개요

 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

 OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.
본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.
예측 csv 파일 제출 (Evaluation) 방식이 아닌 model checkpoint 와 inference.py 를 제출하여 채점하는 방식입니다. (Inference) 상세 제출 방법은 AI Stages 가이드 문서를 참고해 주세요! 대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다.

Input : 글자가 포함된 전체 이미지
Output : bbox 좌표가 포함된 UFO Format (상세 제출 포맷은 평가 방법 탭 및 강의 5강 참조)

제출포맷: UFO

평가방법: DetEval
1) 모든 정답/예측박스들에 대해서 Area Recall, Area Precision을 미리 계산
![image](https://user-images.githubusercontent.com/62556539/206107994-99496577-c39e-4e51-9a8f-6d8fa3449685.png)
2) 모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정
 - 매칭 조건(one-to-one match, one-to-many match, many-to-one match)
![image](https://user-images.githubusercontent.com/62556539/206108078-fee44b19-2a09-4450-b4a3-6b4dde74ec2f.png)
3) 모든 이미지에 대하여 Recall, Precision을 구한 이후, 최종 F1-Score은 모든 이미지 레벨에서 측정 값의 평균으로 측정

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

```


## 프로젝트 수행 환경
모든 실험은 아래의 환경에서 진행되었다.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-PCIE-32GB


## 프로젝트 수행 절차 및 방법

[![image]()



## 프로젝트 수행 결과


## 자체 평가 의견

**잘한 점**
- 

**아쉬운 점:**
- 

**개선할 점:**
- 
---
