# BERT4Rec(2019)

[paper](https://dl.acm.org/doi/pdf/10.1145/3357384.3357895)

Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019, November). BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In _Proceedings of the 28th ACM international conference on information and knowledge management_ (pp. 1441–1450).

pytorch code : [https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)

## ABSTRACT <a href="#abstract" id="abstract"></a>

과거 모델들은 순차적 신경망을 사용하여 왼쪽에서 오른쪽으로 인코딩했지만, 그런 모델들은 최적화됐다고 할 수 없다. 이유1) 단방향 구조는 hidden representation의 가능성을 제한한다. 이유2) 추천에 있어서 순차적 정보를 엄격하게 지키는 것은 실제로는 맞지 않다.

그래서 양방향 모델 BERT4Rec을 제안한다. 이는 왼쪽과 오른쪽 context를 모두 고려할 수 있다.

## 1 INTRODUTION <a href="#1-introdution" id="1-introdution"></a>

유저의 연속적인 행동을 모델링 할 때는 좌, 우방향의 context를 통합하는 것이 중요하다.

**cloze task?**\
왼쪽과 오른쪽 context를 모두 고려하는 것은 정보 유출(information leakage)를 일으킬 수 있다. Target 아이템을 예측해가면서 학습해야하는데, 아이템 서로서로가 간접적으로 타겟 아이템을 이미 참고하고 있기 때문이다.

이 문제를 해결하기 위해 Cloze task라는 것을 적용한다. 단방향 모델에서 순차적으로 다음 아이템을 예측하는 것을 대체하는 것이다.

구체적으로는 일부 아이템을 랜덤으로 masking하여 context를 이용해 masked item을 예측하는 것이다.

## 2 RELATED WORK <a href="#2-related-work" id="2-related-work"></a>

**2.1 General Recommendation**\
초기에 가장 널리 쓰인 모델은 Collaborative Filtering이고, 다양한 CF의 변형들 중에서는 Matrix Factorization이 가장 유명하다. 이는 유저와 아이템을 같은 벡터공간으로 사영하고, 벡터간의 내적으로 선호도를 계산한다.\
딥러닝이 추천시스템에 적용된 첫 번째 모델은 CF에 대한 RBM(Restricted Boltzmann Machines)\[42]이다. 한편 텍스트, 이미지, 청각 피쳐를 아이템 분산표현에 결합한 CF도 있다. MF를 발전시킨 NCF는 내적 대신 MLP로 유저 선호도를 구한다.

**2.2 Sequential Recommendation**\
Markov Chains(MCs), Markov Decision Process(MDPs), Factorizing Personalized Markov Chains(FPMC), GRU4Rec, NARM, Convolutional Sequence Model(Caser), Memory Network 등

**2.3 Attention Mechanism**\
SASRec\[22]은 two-layer Transformer decoder이다. BERT4Rec과 매우 유사하지만, 단방향이라는 단점이 있다.

## 3 BERT4REC <a href="#3-bert4rec" id="3-bert4rec"></a>

### 3.1 Problem Statement <a href="#31-problem-statement" id="31-problem-statement"></a>

![](https://miro.medium.com/max/1400/1\*nmA8KySNOmE3zBXhqT5MpQ.png)

목적 : 인터랙션 히스토리 _Sᵤ\_가 주어졌을 때, \_u\_가 \_nᵤ_+1에 상호작용 할 아이템을 예측하는 것. 모든 아이템에 대해서 확률을 계산한다.

### 3.2 Model Architecture <a href="#32-model-architecture" id="32-model-architecture"></a>

![](https://miro.medium.com/max/1400/1\*vk9noSErbw64XU8Cph0rxQ.png)

BERT4Rec은 양방향 셀프어텐션을 사용한다.

BERT4Rec에는 \_L\_개의 양방향 트랜스포머 레이어가 쌓여있다. 각 레이어는 이전 위치의 모든 레이어들과 정보를 교환한다.

### 3.3 Transformer Layer <a href="#33-transformer-layer" id="33-transformer-layer"></a>

\_t\_길이의 인풋 시퀀스가 주어지면, 위치 \_i\_에 대해, 레이어 \_l\_에서 _**hᵢˡ**_를 동시에 계산한다. _**hᵢˡ**_ ∈ ℝᵈ를 쌓아, 행렬 _**Hˡ**_ ∈ ℝᵗ ˣ ᵈ 로 만든다. 동시에 계산하기 위해서이다. Transformer layer는 멀티헤드 셀프어텐션과 Position-wise Feed-Forward Nerwork로 구성되어있다.

![](https://miro.medium.com/max/712/1\*cEWOTRx6jVoYfhN\_44Szlw.png)

**Multi-Head Self-Attention**\
_**Hˡ**_을 _h_ 서브스페이스로 선형적으로 사영시키고(헤드마다 각각 다르게), h에 어텐션 함수를 병렬적으로 적용하여 output representation을 생성한다. 그리고나서 다시 한 번 더 사영된다.

![](https://miro.medium.com/max/1400/1\*nqO2LtIVD-DP8CJuKkIbcQ.png)

사영에 쓰이는 _**Wᵒ**_는 _d×d 차원이고, 나머지_ _**W**들은 d × d/h 차원이다._ 파라미터들은 layer간에 공유되는 것이 아니다.

![](https://miro.medium.com/max/1400/1\*dEPU8BTGvbz1w-1rHSto3g.png)

query _**Q**_, key _**K**_, value _**V**_는 동일한 행렬 _**Hˡ**로부터 사영되어 만들어진다._

![](https://miro.medium.com/max/1400/1\*5wRXX-Is4JCo9cT8jgRcIQ.png)

**Position-wise Feed-Forward Network**\
셀프어텐션 서브레이어는 선형적인 사영에 베이스를 두고 있다. 모델에 비선형성을 주기 위해서 셀프어텐션 서브레이어의 아웃풋에 포지션와이즈 피드포워드 신경망을 적용한다. 이는 각 포지션(시간적으로, t에 대한)에 대해서 분리적으로 적용한다. 2개의 Affine변환이 GELU(Gaussian Error Linear Unit) 활성화를 사이에 두고 있다.

![](https://miro.medium.com/max/1400/1\*uf4pzxaNOnLa2\_1nDSxTdA.png)

Φ(_x_)는 가우시안 분포의 누적 분포 함수이다. _**W¹**_ ∈ ℝ\_ᵈ ˣ ⁴ᵈ\_ , _**W²**_ ∈ ℝ⁴\_ᵈ ˣ ᵈ\_ , _**b¹**_ ∈ ℝ\_⁴ᵈ\_ and _**b²**_ ∈ ℝ\_ᵈ\_. 이 파라미터들은 레이어마다 다르다. 기본 ReLu 활성화보다, smoother GELU 활성화를 사용한다.

**Stacking Transformer Layer**\
residual connection, 각 서브레이어 결과에 dropout 적용

![](https://miro.medium.com/max/1400/1\*\_Wqrq\_5lKqy5ihRz0FTzdg.png)

transformer layer

### 3.4 Embedding Layer <a href="#34-embedding-layer" id="34-embedding-layer"></a>

트랜스포머 레이어의 인풋인 _**hᵢ⁰**_는 아이템 벡터 _**vᵢ**_에 포지셔널 인코딩을 더해서 구해짐.

_**hᵢ⁰ = vᵢ + pᵢ**_

아이템벡터 _**vᵢ**_는 아이템 _**vᵢ**_에 대한 \_d\_차원 임베딩이다.

포지셔널 임베딩으로 인해 인풋 시퀀스를 maximum sentence length N에 맞추어 최근 N개만 잘라서 사용해야한다.

### 3.5 Output Layer <a href="#35-output-layer" id="35-output-layer"></a>

**H^L을 얻고난 후,**

![](https://miro.medium.com/max/1400/1\*nBUNuwlfXGWlQMM7D-6ujA.png)

오버피팅을 막고 모델 사이즈를 줄이기 위해 인풋에서 썼던 E를 다시 쓴다.

### 3.6 Model Learning <a href="#36-model-learning_1" id="36-model-learning_1"></a>

**Train**\
단방향 모델은 계속 다음 \_t\_의 아이템을 예측하며 학습하지만, 양방향 모델은 앞에 있는 아이템을 이미 알고 있으므로 leakage가 발생한다.

1\~_t_-1까지만 떼어내서 이들을 양방향으로 인코딩한 다음 마지막 아이템을 타겟으로 예측하는 방법도 있지만, 시간과 리소스면에서 안좋다고 함.

그래서 Masek 언어모델의 방법을 씀. 임의의 portion ρ만큼의 아이템을 \[mask] 토큰으로 대체하고나서, 좌, 우 문맥에 의존해서 원래 id를 예측한다.

마스크된 아이템의 \_h\_도 소프트맥스로 들어가고, negative log-likelihood로 손실을 구함.

![](https://miro.medium.com/max/1400/1\*22nO1hqouDOC7b3s6VUMGw.png)

_**Sᵤ**_ : user behavior history\
_**Sᵤ’**_ : _**Sᵤ**_의 masked version\
_**Sᵤᵐ**_ : random masked items\
_**vₘ**_ : masked item\
_**vₘ\***_ : true item for the masked item vₘ

**Test**\
Sequential Recommendation은 다음에 등장할 아이템을 예측하는 것인데, 이 모델은 마스크된 아이템을 예측하다보니 미스매치가 존재한다. 이를 해결하기 위해 user behavior의 마지막에 \[mask] 토큰을 추가하고, 마지막 hidden representaion(mask 토큰꺼)으로 다음 아이템을 예측하도록 한다. 거기에 더욱 Sequential recommenation 태스크처럼(마지막 아이템 예측하는 것) 하기 위해서, 학습할 때 마지막 아이템 하나만 마스크한 샘플을 추가해준다.

\[22] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In Proceedings of ICDM. 197–206

\[42] Ruslan Salakhutdinov, Andriy Mnih, and Geoffrey Hinton. 2007. Restricted Boltzmann Machines for Collaborative Filtering. In Proceedings of ICML. 791– 798.



