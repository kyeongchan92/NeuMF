---
description: >-
  차루 아가르왈의 '추천시스템 (recommender systems)' 책의
  내용입니다(https://product.kyobobook.co.kr/detail/S000001805083).
---

# 이웃 기반 협업 필터링

## 이웃 기반 협업 필터링(=메모리 기반 알고리즘)

1.  유저 기반 협업 필터링

    타겟 유저 $$A$$에게 추천을 제공해주기 위해 $$A$$와 유사한 유저들(Peer group)이 아이템에 준 평점에 가중평균을 적용해서 구한다.



    <figure><img src="../.gitbook/assets/image (1) (2).png" alt=""><figcaption></figcaption></figure>


2.  아이템 기반 협업 필터링

    타겟 아이템 $$B$$에 대한 추천을 만든다. 우선 $$B$$와 가장 유사한 아이템 집합 S를 결정한다. 특정 유저 $$A$$의 $$B$$에 대한 평점을 계산하기 위해서는 $$A$$가 $$S$$에 매긴 평점이 필요하다. 이 평점들의 가중평균이 추천 결과가 된다.

## 평점 행렬이란

평점 행렬은 $$R$$이라 표현하고, $$m$$명의 사용자와 $$n$$개의 아이템을 가지고 있는 $$m\times n$$ 행렬이다. 평점 중 일부를 학습데이터로 사용하고, 나머지는 테스트 데이터로 사용한다\[22].

<figure><img src="../.gitbook/assets/image (13).png" alt=""><figcaption></figcaption></figure>

## 이웃 기반 방법론의 평점 예측

이웃 기반 방법론은 이름 그대로 유저(아이템) 자기 자신과의 이웃, 즉 유사한 유저(아이템)을 이용해 문제를 푼다. 그래서 유저(아이템) 서로 간의 유사도를 구하는 것이 기본적이면서 중요한 과정이 된다.

<figure><img src="../.gitbook/assets/image (4) (3).png" alt=""><figcaption><p>?에 들어갈 숫자는 무엇일까. 초극단적 예시이긴 하지만..</p></figcaption></figure>

### 사용자 기반 이웃 모델

이 접근법에서는 타겟 유저와 유사한 유저를 찾아야한다. 두 사용자 $$u$$와 $$v$$의 평점 벡터 간의 유사도 $$\text{Sim}(u, v)$$를 구하는 방법 중 하나는 **피어슨 상관계수**이다. 피어슨 상관계수를 계산할 때는 $$u$$와 $$v$$ 둘 모두가 평가한 아이템에 대해서만 계산된다. 일단 각 사용자 $$u$$에 대한 평점 평균을 구해야한다.

<figure><img src="../.gitbook/assets/image (16).png" alt=""><figcaption><p>equation 2.1</p></figcaption></figure>

그리고 $$u$$와 $$v$$의 행(유저)간의 피어슨 상관계수는 다음과 같다.

<figure><img src="../.gitbook/assets/image (4).png" alt=""><figcaption><p>equation 2.2</p></figcaption></figure>

## 사용자 기반 알고리즘의 예시

<figure><img src="../.gitbook/assets/image (15).png" alt=""><figcaption><p>Table 2.1</p></figcaption></figure>

Table 2.1의 예제를 생각해보자. 다섯 명의 유저 0 \~ 4와 여섯 개의 아이템 0 \~ 5가 있다. 평점은 1 \~ 7의 범주를 갖는다고 하자. 타겟 유저가 **유저 2**라고 하자. 우리는 유저 2의 아이템 0과 5에 대한 예측, $$\hat{r}_{20}$$, $$\hat{r}_{25}$$를 계산해야한다.

가장 먼저 해야할 것은 유저 2와 다른 모든 유저들 간의 유사도 계산이다. 예를 들면 유저 0과 유저 2의 피어슨 상관계수는 다음과 같다.

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

유저 2와 다른 모든 유저와의 비어슨 유사도는 Table 2.1의 오른쪽 표와 같다. 유저 3과 가장 가까운 상위 2명은 **유저 0**과 **유저 1**이다. $$\hat{r}_{20}$$, $$\hat{r}_{25}$$를 계산하기 위하여, 유저 0과 유저 1의 평점을 사용한다.

<figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption></figcaption></figure>

