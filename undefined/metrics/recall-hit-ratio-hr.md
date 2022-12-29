# 추천에서의 Recall (Hit Ratio, HR)

## 일반적인 Recall

### train test split

<figure><img src="https://wikidocs.net/images/page/180451/%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%86%E1%85%B5%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC.png" alt=""><figcaption></figcaption></figure>

그림과 같이 어떤 유저가 사용한 아이템과 아직 사용하지 않은 아이템이 있다고 하자. 실제 데이터에서는 사용 아이템보다는 미사용 아이템이이 **아주 훨씬 엄청 되게 심하게** 많겠지?

<figure><img src="../../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

우선적으로, 사용한 아이템을 train과 test로 나누게 될 것이다. 아래 그림처럼 학습데이터셋으로 A, C, E라는 아이템이, 테스트 셋으로 I, K, G라는 아이템이 되었다고 하자.

학습이 진행될 것이다. 학습이 끝났다 치자!

### 계산

<figure><img src="../../.gitbook/assets/image (3) (2) (1).png" alt=""><figcaption></figcaption></figure>

테스트의 시간이 왔다. 테스트 대상이 되는 아이템은 1) interaction이 1인데 테스트셋으로 분류된 아이템과 2) 미사용한 아이템이다.

k를 3으로 하고 Recall@3을 구해보자. 모델이 사용확률 높은 순서대로 정렬 했을 때 top 3가 K, G, B 라고 해보자. Recall에서는 **순서는 상관없다!** 이들이 선정되었다는 사실까지만 필요하다.

<figure><img src="../../.gitbook/assets/image (7) (2).png" alt=""><figcaption></figcaption></figure>

아까 **테스트로 분류된 사용한 아이템**들, I, K, G을 '정답셋'이라고 하자. Recall은 예측한 아이템들과 정답셋의 교집합 개수가 분자로, k가 분모로 계산된 값이다.

## Sequential 추천에서의 Recall



