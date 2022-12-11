# 이웃 기반 협업 필터링

## 이웃 기반 협업 필터링(=메모리 기반 알고리즘)

1.  유저 기반 협업 필터링

    타겟 유저 A에게 추천을 제공해주기 위해 A와 유사한 유저들(Peer group)이 아이템에 준 평점에 가중평균을 적용해서 구한다.



    <figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>


2.  아이템 기반 협업 필터링

    타겟 아이템 B에 대한 추천을 만든다. 우선 B와 가장 유사한 아이템 집합 S를 결정한다. 특정 유저 A의 B에 대한 평점을 계산하기 위해서는 A가 S에 매긴 평점이 필요하다. 이 평점들의 가중평균이 추천 결과가 된다.

## 평점 행렬이란

평점 행렬은 R이라 표현하고, m명의 사용자와 n개의 아이템을 가지고 있는 mXn 행렬이다. 평점 중 일부를 학습데이터로 사용하고, 일부는 테스트 데이터로 사용한다\[22].

<figure><img src="../.gitbook/assets/image (13).png" alt=""><figcaption></figcaption></figure>


