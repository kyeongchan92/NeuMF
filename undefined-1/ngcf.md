# NGCF



[Neural graph collaborative filtering(2019)](https://dl.acm.org/doi/abs/10.1145/3331184.3331267?casa\_token=A9-7uYyz9tYAAAAA:-ek-mqln7SYN4f9NwrWZc9XB9tXXT3LaUoLMpAYx0qoyVSuxawLHMi\_uGLWsGH43V0U7IKD-2Pg)

pytorch 코드 : [huangtinglin/NGCF-PyTorch](https://github.com/huangtinglin/NGCF-PyTorch)

1. 임베딩 레이어
2. 전파 레이어
   1. First-order propagation
   2. High-order propagation
3. 예측 레이어

## 1. 임베딩 레이어 <a href="#1" id="1"></a>

NCF같은 기존 모델은 ID임베딩을 바로 interaction layer로 전달하여 예측 스코어를 계산합니다. 하지만 NGCF는 유저-아이템 상호작용 그래프에 propagation하여 임베딩을 정제합니다. 이렇게 하면 collaborative signal을 임베딩에 명시적으로 반영할 수 있습니다.

<figure><img src="../.gitbook/assets/image (3) (1).png" alt=""><figcaption></figcaption></figure>

## 2. 전파 레이어 <a href="#2" id="2"></a>

GNN의 message-passing 아키텍쳐를 이용하여 그래프 구조를 따라 CF 시그널을 잡아내고 유저-아이템 임베딩을 정제합니다. 우선 하나의 레이어에서 어떻게 전파가 이루어지는지 확인하고, 다중 레이어로 확장해봅시다.

### 2.1 First-order propagation <a href="#21-first-order-propagation" id="21-first-order-propagation"></a>

직관적으로 생각해봤을 때, 어느 유저에 의해 사용된(클릭 등) 아이템은 해당 유저의 선호도를 구성할 수 있습니다. 마찬가지로, 특정 아이템을 사용한 유저들은 해당 아이템의 특징을 구성하고 있습니다. 또한 두 아이템의 유사도를 측정하는데 사용될 수도 있을 겁니다. 이러한 가정을 기반으로, 상호작용한 유저와 아이템 사이에 임베딩 전파를 수행합니다. 이 때, message construction과 message aggregation이라는 개념이 사용됩니다.

**Message Construction** 서로 연결된 유저-아이템 쌍 $$(u, i)$$에 대하여, $$i$$에서 $$u$$로의 message를 다음과 같이 정의합니다.

$$
m_{u \leftarrow i} = f(e_i, e_u, p_{ui})
$$

$$m_{u \leftarrow i}$$는 메시지 임베딩, 즉 전파되는 정보를 의미합니다. $$f(\cdot)$$는 메시지 인코딩 함수라고 하며, 임베딩 $$e_i$$와 $$e_u$$, 그리고 의 전파에 대한 감쇠인자(decay factor)를 조정하기 위한 계수 를 인풋으로 받습니다.

본 논문에서는 f(⋅)을 다음과 같이 정의합니다:

mu←i=1|Nu||Ni|(W1ei+W2(ei⊙eu))

W1,W2∈Rd′×d는 학습 가능한 가중치 행렬입니다. 이 행렬은 전파라는 것이 유용한 정보를 증류(distill)하게 합니다. 전통적인 Graph convolutional networks에서는 오직 ei의 공헌만을 고려했지만, 본 논문에서는 ei와 eu사이의 상호작용을 메시지에 추가로 인코딩합니다. ⊙은 element-wise product를 의미합니다. 이렇게 하면 메시지를 ei와 eu사이의 유사도에 의존적이게 만듭니다. 예를 들면, 유사한 아이템들로부터는 더 많은 메시지를 받는 것입니다. 이는 모델의 표현(representation) 능력을 더 좋게 할 뿐만 아니라, 추천의 성능도 향상시킵니다.

![message\_construction](https://wikidocs.net/images/page/176711/image.png)

Graph convolutional network처럼, 여기서는 graph Laplacian norm pui=1|Nu||Ni|을 설정합니다. 여기서 Nu과 Ni는 각각 유저 u와 아이템 i의 first-hop 이웃을 나타냅니다. representation 학습의 관점에서 pui는 아이템 i가 유저 u의 선호도에 얼만큼 공헌했는지를 의미합니다. 아이템 i를 소비한 유저가 많다면, 그 중 한 유저에게 보내는 메시지는 영향력이 적겠죠? 유저 입장에서도 마찬가지입니다. 유저 u가 아주 많은 아이템을 소비했다면, 그 중 특정 아이템 i 사이의 메시지 mu←i의 영향력이 줄어드는 것이 맞겠죠.

![hop](https://wikidocs.net/images/page/176711/hop.png)

한편 메시지 전달 관점에서 pui는 감쇠 인자로도 볼 수 있습니다. 즉, 전파되는 메시지 경로의 길이가 길어짐에 따라 공헌도가 감소해야 한다는 것을 반영합니다.

**Message Aggregation** 메시지를 공식화하여 벡터로 나타내기까지 했으면, 이제 유저 u의 이웃으로부터 전파되는 메시지들을 결합함으로써 u의 표현을 정제합니다. 구체적으로, Aggregation의 함수는 다음과 같습니다:

eu(1)=LeakyReLU(mu←u+∑i∈Numu←i)

eu(1)은 유저 u의 표현을 나타내는데, 위첨자 (1)은 첫 번째 전파 레이어의 결과라는 의미입니다. 활성화 함수로 쓰이는 LeakyReLU는 양수가 들어오는 경우 그대로 통과시키고 음수는 0.01을 곱한 값을 내놓는 함수입니다. 주목할 점은 이웃 Nu로부터 전파된 메시지들 뿐만 아니라, 유저 u의 self-connection mu←u=W1eu을 고려했다는 것입니다. mu←u는 원본 특징(original feature)의 정보를 보유하고 있습니다. W1은 mu←i에서 쓰였던 W1과 동일한 가중치 행렬입니다. 같은 행렬이 다시 한 번 쓰이는 것입니다. 유사하게, 아이템 i에 대한 표현 ei(1)도 얻을 수 있습니다. 이 때는 연결된 유저들로부터의 정보를 전파하면 됩니다.

![leakyReLU](https://wikidocs.net/images/page/176711/leakyrelu.png) ![message\_aggregation](https://wikidocs.net/images/page/176711/message\_aggregation.png)

요약하자면 전파 레이어의 장점은 1차원의 연결 정보를 이용하여 유저와 아이템 각각의 표현을 결부시킨다는 사실에 있습니다.

### 2.2 High-order Propagation <a href="#22-high-order-propagation" id="22-high-order-propagation"></a>

1차원 연결 모델링을 통해 얻은 표현부터 시작하여, 임베딩 전파 레이어를 더 쌓아 고차원 연결의 정보를 얻을 수 있습니다.

l개의 전파 레이어를 쌓음으로써, 유저(혹은 아이템)은 자신의 l-hop으로부터 전파된 메시지를 받을 수 있습니다. Figure 2를 통해 볼 수 있는 것처럼, l번째 스텝에서는 유저 u의 표현이 재귀적으로 다음과 같이 공식화됩니다:

eu(l)=LeakyReLU(mu←u(l)+∑i∈Numu←i(l))

그렇다면 l-hop으로부터 오는 메시지는 어떻게 공식화될까요? 다음과 같습니다:

mu←i(l)=pui(W1(l)ei(l−1)+W2(l)(ei(l−1)⊙eu(l−1)))

mu←u(l)=W1(l)eu(l−1)

W1(l),W2(l)∈Rdl×dl−1은 학습가능한 변형행렬이고, dl은 변형 후 사이즈입니다. eil−1은 이전 스텝으로부터 생성된 아이템 표현입니다. 이 표현은 (l−1)-hop 이웃으로부터의 메시지를 기억하고 있습니다. Figure 3에서 볼 수 있듯이, u1←i2←u2←i4와 같은 collaborative signal이 전파 과정에 의해 학습될 수 있습니다. i4로부터의 메시지는 eu1(3)에 명시적으로 인코딩됩니다.

![propagaion1](https://wikidocs.net/images/page/176711/image\_4.png)

u1은 어떻게 3-hop의 위치에 있는 i4의 정보까지 포함할 수 있는지 다시 한 번 천천히 살펴봅시다. 위와 같은 그래프(Figure 3과 동일)가 있다고 할 때, 가장 첫 작업(첫 번째 레이어)에서 u2는 i4로부터 오는 메시지를 받아 업데이트 될 것입니다. 즉, eu2(1)은 i4 정보를 포함하게 됩니다. 물론 이 단계에서 모든 아이템과 유저 임베딩이 동일하게 업데이트 될 것입니다.

![propagaion2](https://wikidocs.net/images/page/176711/image\_5.png)

두 번째 작업(레이어)에서는 i2 주변 유저들로부터 메시지를 받아 업데이트 될 것입니다. 그런데 이 유저 중 한 명인 u2는 이전 레이어에서 i4의 정보를 포함하게 됐었죠. 그래서 두 번째 레이어까지 업데이트를 마친 결과, 결과적으로 i4의 정보를 포함한 u2, u2의 정보를 포함한 i2가 되었습니다.

![propagaion3](https://wikidocs.net/images/page/176711/image\_6.png)

그 다음단계에서 u1은 마찬가지로 주변 아이템 i1,i2,i3로부터 메시지를 맞아 업데이트 되겠죠. 근데 i2는 i4의 정보까지 포함하고 있었네요. 따라서 이번 레이어에서의 작업을 마치면 u1은 i4의 정보까지 포함하게 될 것입니다.

### Propagation Rule in Matrix Form <a href="#propagation-rule-in-matrix-form" id="propagation-rule-in-matrix-form"></a>

임베딩 전파 및 배치 수행을 위해서는, 실제로는 행렬을 이용해 연산을 수행합니다. 레이어 단위로 수행되는 전파는 아래 수식을 따릅니다.

E(l)=LeakyReLU((L+I)E(l−1)W1(l−1)+LE(l−1)⊙E(l−1)W2(l))

E의 shape는 E(l)∈R(N+M)×dl 입니다. N은 유저의 수, M은 아이템의 수입니다.

![RtoA](https://wikidocs.net/images/page/176711/RtoA.png)

L은 유저-아이템 그래프에 대한 Laplacian 행렬이며 다음과 같이 정의됩니다:

L=D−12AD−12andA=\[0RR⊤0]

R∈RN×M은 유저-아이템 상호작용 행렬이며, 0는 영행렬입니다. A는 인접행렬이며, D는 대각 degree 행렬입니다. D의 t-th 대각 요소는 Dtt=|Nt|입니다. 즉, 해당 유저 또는 아이템의 1-hop 유저의 수입니다. Lui=1/|Nu||Ni|가 됩니다. 이 수는 message construction에서 봤던 그 계수와 동일합니다.

행렬 계산 형태로 propagation을 수행함으로써, 우리는 모든 유저와 아이템에 대한 표현을 동시에 효과적으로 업데이트할 수 있습니다. Graph convolutional network에서는 보통 노드 샘플링 과정이 있는데, 이렇게 행렬 계산을 함으로써 이 과정도 없어집니다.

### 번외) L=D−12AD−12가 등장한 이유 <a href="#mathcall-mathbfd-frac12mathbfamathbfd-frac12" id="mathcall-mathbfd-frac12mathbfamathbfd-frac12"></a>

L=D−12AD−12는 어떤 노드가 주변 노드의 정보를 aggregation할 때 그 message의 decay factor가 들어있는 행렬입니다.

![laplacian1](https://wikidocs.net/images/page/176711/laplacian1.png)

논문에 나오는 간단한 그래프를 예로 들어봅시다. 3명의 유저와 5개의 아이템이 있습니다. 본 논문에서 정의한 인접행렬 A에는 R이라는 인터랙션 행렬이 블록행렬 형태로 들어가 있습니다.

Laplacian 계산을 해봅시다.

![laplacian2](https://wikidocs.net/images/page/176711/laplacian2.png) ![laplacian3](https://wikidocs.net/images/page/176711/laplacian3.png) 인접행렬에서 1이었던 자리는 유저와 아이템의 인터랙션이 존재했음을 의미합니다. 앞뒤로 각 노드의 degree의 1|N|을 곱해주니까, 1이 있던 자리에는 유저의 이웃 수와 아이템의 이웃 수를 각각 루트를 씌워 역수를 취한 후 두 수를 곱한 수가 되었습니다.

이 숫자의 의미는 앞 장에서 설명했던 것처럼, 인기가 많은(다른 유저와 인터랙션이 많은) 아이템으로부터의 메시지의 정보 크기는 그 이웃 수만큼 줄인다는 것입니다.

![laplacian4](https://wikidocs.net/images/page/176711/laplacian4.png)

그래서 (L+I)E를 수행하게 되면, 연결된 주변 노드로부터 decay factor가 곱해진 정보를 받아들여 더해지게 됩니다.

\





