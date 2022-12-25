# NARM(Neural Attentive session-based recommendation)

[Li, J., Ren, P., Chen, Z., Ren, Z., Lian, T., & Ma, J. (2017, November). Neural attentive session-based recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (pp. 1419-1428).](https://arxiv.org/pdf/1711.04725.pdf)

NARM은 인코더-디코더 구조를 따른다. NARM은 세션을 hidden representation으로 만들고 이를 이용해 예측을 수행한다.

![](https://miro.medium.com/max/1400/1\*VnSRNys0x3aN7ERVUooUmQ.png)

인코더는 t개의 아이템 시퀀스를 입력으로 받아 t개의 hidden representations h=\[h₁, h₂, …, hₜ₋₁, hₜ]를 만든다. 이 h는 αₜ와 함께 Session Feature Generator로 들어가서 cₜ라는 representation을 만든다. 마지막으로 cₜ는 U라는 행렬에 의하여 모든 아이템에 대한 랭킹 리스트 y=\[y₁, y₂, …, yₘ₋₁, yₘ]로 변환된다.

αₜ는 hidden representation의 어떤 부분이 강조되고, 또는 무시되어야 하는 지를 결정해주는 역할을 한다.

이 모델은 “유저의 시퀀셜한 행동(현재 세션에 대한 전반적인 정보)”과 “현재 세션의 목적(현재 세션에서 어떤 아이템이 예측에 핵심적인지)” 둘 다 학습한다는 것이 특징이다.

## 인코더 <a href="#_1" id="_1"></a>

인코더는 Global 인코더와 Local 인코더로 나뉜다.

![](https://miro.medium.com/max/1400/1\*F\_Uatc3v2XrffTqAIfBt\_w.png)

인코더는 Global 인코더와 Local 인코더로 나뉜다.

## Global Encoder <a href="#global-encoder" id="global-encoder"></a>

Global Encoder는 GRU를 통과시킨 후 마지막 hidden state만 아웃풋으로 내놓는다(파란 벡터). 이는 hₜᵍ이기도 하면서 이 논문에서는 cₜᵍ라고도 부른다. 같은 벡터를 지칭한다. 이 cₜᵍ라는 컨텍스트 벡터에는 “**유저의 시퀀셜 행동에 대한 정보”**가 들어 있다고 할 수 있다.

## Local Encoder <a href="#local-encoder" id="local-encoder"></a>

Local Encoder는 일단 각 시간마다 모든 hidden state를 뽑아낸다(연두색). 이 h들에는 각각의 시간에 해당하는 아이템에 대한 정보가 많이 들어 있다고 볼 수 있다. 예를 들면 h₅ˡ라는 hidden state에는 x₅라는 아이템에 대한 정보가 많이 들어있을 것이다.

Local Encoder에서는 시퀀스에 존재하는 아이템 중 어떤 아이템에 주목해야 하는지 알고 싶어한다. 마지막 hidden state인 hₜˡ을 그대로 다음 단계로 넘겨주지 않는다. 어떤 아이템이 예측에 핵심적인지 알기 위하여 h들과 hₜˡ을 이용해 추가적인 Attention 메커니즘 계산 과정이 필요하다.

![](https://miro.medium.com/max/1400/1\*Ph7h6EmNTHrzDVbDlxiUvA.png)

첫 번째로, hidden state들과 hₜˡ 사이의 어떤 연산(구름 표시)을 한다. 이 연산은 유사도를 구하는 과정이라고 이해해도 된다. 구름 안의 연산은 아래 그림과 같다. 사실 내적을 해도 되는 것 같은데, 이 논문에서는 아래처럼 latent space로 보내 연산을 한다.

![](https://miro.medium.com/max/1400/1\*STMaESpYCBFM7Nq5VObIRQ.png)

여기까지 하면 α까지 구한 것이다. 본 논문에서는 α를 weighted factor라고 부른다. α는 현재 세션 아이템의 개수인 t개 만큼 만들어진다. 당연한 것이 Local Encoder가 아이템 개수 만큼의 hidden state를 만들었을테니 말이다. 즉 αₜ₁, αₜ₂, …, αₜₜ가 생긴다. 이 α들은 각 아이템의 가중치를 의미한다.

![](https://miro.medium.com/max/1400/1\*xda51DU4DrUR2FOmT8sfEA.png)

이제 이 weighted factor를 각 hidden states에 곱하고, 다 더한다. 이렇게 되면 α가 큰 위치의 h벡터가 많은 비중을 차지한다. α가 큰 hidden state는 마지막 hₜ와 유사하다는 것이다. 다른 말로 α가 큰 hidden state는 context vector를 많이 설명하게 된다.

이렇게 다 더하고 나면 context vector인 cₜˡ이 만들어진다. 이 벡터에는 각 아이템의 hidden representation이라고 할 수 있는 h들이 어떠한 비율로 녹아 들어가 있을 것이다. 즉, 이 벡터는 유저가 세션 내 아이템들 중에서 어떤 아이템에 주목하고 있는지를 나타낸다. “**유저의 주 목적”**을 표현하고 있다고 할 수 있다.

![](https://miro.medium.com/max/1046/1\*U7cltwrTtt8jmDWZoDPe2g.png)

Global Encoder에서 구한 cₜᵍ와 Local Encoder에서 구한 cₜˡ을 concat하여 cₜ를 만든다.

자 이렇게 해서 cₜ를 얻었다. 이 벡터는 현재 세션을 표현하고 있다. 이제 해야 할 것은 이 cₜ(현재 세션의 압축된 정보)로부터 각 아이템이 등장할 확률을 계산하는 것이다.

처음 그림과 잘 맞게 설명한 것인지 모르겠지만, 여기까지가 인코더+Session Feature Generator를 끝낸 것이다.

### 디코더 <a href="#_2" id="_2"></a>

![](https://miro.medium.com/max/1400/1\*t220v3e7pFgbyrH\_PokvHw.png)

위 그림은 cₜ로부터 각 아이템의 Score를 계산하는 아주 간단한 연산이다. 각 아이템의 스코어를 Sᵢ라고 하자. 앞에서 구한 cₜ의 차원을 |H|라고 하자. 아이템 임베딩 벡터가 |D|차원이라고 하자. 위 그림에서 아이템 i의 임베딩 벡터는 embᵢ이다. 그리고 |D|는 3이다.

논문에서는 이 과정을 bi-linear decoding scheme이라고 부른다. 이 과정이 끝나면 각 아이템은 개별적으로 Sᵢ를 얻게 된다. 이를 마지막으로 Softmax 계층을 통과시켜 모든 아이템에 대한 확률 값으로 바꾸면 연산이 마무리된다.

손실은 Cross-Entropy Loss를 사용한다.

![](https://miro.medium.com/max/1400/1\*DYloEGsvjRyrWG-v1vyLTg.png)

일반적인 RNN은 디코딩 과정에서 fully connected layer를 쓴다. 만약 이 경우에 fully connected layer를 쓴다면, |H|차원(cₜ의 차원)의 벡터를 |N|(아이템 수) 차원으로 변환하는 Affine 변환이 필요할 것이다. 그럼 해당 layer는 |H|×|N| 행렬이 될 것이다. 그러나 아이템 수는 보통 수 만 개가 된다. 이 논문에서 사용하는 데이터의 아이템 수도 16,000개, 40,000개 정도 된다. 그럼 파라미터가 너무 많아진다.

논문에서는 위 그림과 같은 bi-linear decoding scheme 방법이 파라미터의 수도 줄일 수 있을 뿐만 아니라 성능도 더 높다고 한다.

끝!
