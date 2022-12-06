# NGCF(Neural Graph Collaborative Filtering)



[Neural graph collaborative filtering(2019)](https://dl.acm.org/doi/abs/10.1145/3331184.3331267?casa\_token=A9-7uYyz9tYAAAAA:-ek-mqln7SYN4f9NwrWZc9XB9tXXT3LaUoLMpAYx0qoyVSuxawLHMi\_uGLWsGH43V0U7IKD-2Pg) 논문의 Methodology 부분을 다룹니다. 논문에서 Methodology에 대한 설명은 크게 세 부분으로 구성되어 있습니다.

pytorch 코드 : [huangtinglin/NGCF-PyTorch](https://github.com/huangtinglin/NGCF-PyTorch)

1. 임베딩 레이어
2. 전파 레이어
   1. First-order propagation
   2. High-order propagation
3. 예측 레이어

## 1. 임베딩 레이어 <a href="#1" id="1"></a>

NCF같은 기존 모델은 ID임베딩을 바로 interaction layer로 전달하여 예측 스코어를 계산합니다. 하지만 NGCF는 유저-아이템 상호작용 그래프에 propagation하여 임베딩을 정제합니다. 이렇게 하면 collaborative signal을 임베딩에 명시적으로 반영할 수 있습니다.

<figure><img src="../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

## 2. 전파 레이어 <a href="#2" id="2"></a>

GNN의 message-passing 아키텍쳐를 이용하여 그래프 구조를 따라 CF 시그널을 잡아내고 유저-아이템 임베딩을 정제합니다. 우선 하나의 레이어에서 어떻게 전파가 이루어지는지 확인하고, 다중 레이어로 확장해봅시다.

### 2.1 First-order propagation <a href="#21-first-order-propagation" id="21-first-order-propagation"></a>

직관적으로 생각해봤을 때, 어느 유저에 의해 사용된(클릭 등) 아이템은 해당 유저의 선호도를 구성할 수 있습니다. 마찬가지로, 특정 아이템을 사용한 유저들은 해당 아이템의 특징을 구성하고 있습니다. 또한 두 아이템의 유사도를 측정하는데 사용될 수도 있을 겁니다. 이러한 가정을 기반으로, 상호작용한 유저와 아이템 사이에 임베딩 전파를 수행합니다. 이 때, message construction과 message aggregation이라는 개념이 사용됩니다.

**Message Construction** 서로 연결된 유저-아이템 쌍 $$(u, i)$$에 대하여, $$i$$에서 $$u$$로의 message를 다음과 같이 정의합니다.

$$
m_{u \leftarrow i} = f(e_i, e_u, p_{ui})
$$

$$m_{u \leftarrow i}$$는 메시지 임베딩, 즉 전파되는 정보를 의미합니다. $$f(\cdot)$$는 메시지 인코딩 함수라고 하며, 임베딩 $$e_i$$와 $$e_u$$, 그리고 의 전파에 대한 감쇠인자(decay factor)를 조정하기 위한 계수 를 인풋으로 받습니다.





