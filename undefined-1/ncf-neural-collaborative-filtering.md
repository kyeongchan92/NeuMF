# NCF(Neural Collaborative Filtering)

[He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)

## 2. PRELIMINARIES <a href="#21-learning-from-implicit-data" id="21-learning-from-implicit-data"></a>

### 2.1 Learning from Implicit Data <a href="#21-learning-from-implicit-data" id="21-learning-from-implicit-data"></a>

$$M$$명의 유저와 $$N$$개의 아이템이 있다고 하자. 유저-아이템 상호작용 행렬 $$\mathbf{Y} \in \mathbb{R}^{M \times N}$$는 implicit feedback에 의해 다음과 같이 정의된다.

$$
y_{ui}=
\begin{cases}
1, & \text{if interaction (user }u, \text{item }i \text{) is observed}; \\
0, & \text{otherwise}
\end{cases}
$$

### 2.2 Matrix Factorization <a href="#31-general-framework" id="31-general-framework"></a>

MF는 유저와 아이템을 latent feature의 실수 벡터와 연관짓는다. $$\mathbf{p}_u$$와 $$\mathbf{q}_i$$가 유저 $$u$$와 아이템 $$i$$를 나타낸다고 해보자. MF는 상호작용 $$y_{ui}$$를 $$\mathbf{p}_u$$와 $$\mathbf{q}_i$$의 내적으로서 계산한다.

$$
\hat{y}_{ui}=f(u,i|\mathbf{p}_u, \mathbf{q}_i)=\mathbf{p}_u^T\mathbf{q}_i=\sum_{k=1}^{K}p_{uk}q_{ik}
$$

$$K$$는 latent space의 차원을 나타낸다. MF는 latent space의 각 차원이 각각의 유저에 대해 독립적이라고 가정하고, 그들을 같은 가중치로 선형적으로 결합한다. 그렇기 때문에 MF는 latent factor의 선형 모델로 여겨진다.

<figure><img src="../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>

Figure1은 내적이 어떻게 MF의 표현력을 제안할 수 있는지를 보여준다. 이를 이해하기 위해 우선적으로 두 가지만 얘기하고 넘어가자. 첫 번째, MF는 유저와 아이템을 같은 latent space로 매핑하기 때문에 두 유저간의 유사도 또한 내적으로 계산될 수 있다(같은 방법으로는 latent vector 사이의 코사인 각도). 두 번째로, 일반화 손실 없이, 우리는 자카드 계수를 MF가 회복시켜야 하는 두 유저간의 ground-truth 유사도로서 사용한다.

Figure 1a의 상위 3개의 row를 보자. $$s_{23}(0.66) > s_{12}(0.5) > s_{13}(0.4)$$를 계산하기는 쉽다. 그렇게 해서, latent space에서의 $$\mathbf{p}_1$$과 $$\mathbf{p}_2$$, 그리고 $$\mathbf{p}_3$$의 기하학적 관계는 Figure 1b처럼 그릴 수 있다. 자 이제 새로운 유저 $$u_4$$를 고려해보자. $$s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)$$를 계산할 수 있는데, 이는 $$u_4$$가 $$u_1$$과 가장 유사하고, 그 다음으로 $$u_3$$와, 그 다음으로 $$u_2$$ 순으로 유사하다는 뜻이다. 그러나, 만약 MF 모델이 $$\mathbf{p}_4$$를 $$\mathbf{p}_1$$에 가장 가깝게 놓는다면 $$\mathbf{p}_4$$가 $$\mathbf{p}_3$$보다 $$\mathbf{p}_2$$에 더 가까워지게 되어 큰 랭킹 손실이 생길 것이다.

**위 예시는 MF가 복잡한 유저-아이템 상호작용을 저차원의 latent space에서 간단하고 고정적인 내적으로 계산하려 할 때 발생할 수 있는 한계를 보여준다.** 한 가지 해결 방법은 $$K$$를 늘리는 것이다. 그러나 이는 sparse한 상황에서는 역으로 모델의 일반화 성능을 저하시킬 수 있다(e.g., 오버피팅). 본 논문에서는 Deep Neural Network를 이용하여 상호작용 함수를 학습시킴으로써 이러한 한계를 다룬다.

## 3. NEURAL COLLABORATIVE FILTERING

### 3.1 General Framework

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption></figcaption></figure>

유저 $$u$$와 아이템 $$i$$의 상호작용을 $$y_{ui}$$라고 하자. 그림 2에서 가장 아래 쪽에 있는 인풋 레이어는 두 개의 피쳐 벡터들로 구성되어있다. 이를 각각 유저 $$u$$와 아이템 $$i$$를 가리키는 피쳐벡터, $$\mathbf{v}_u^U$$와 $$\mathbf{v}_i^I$$라고 하자. 이 피쳐벡터들은 경우에 따라 커스터마이징이 가능하다. 즉, 유저나 아이템의 side-information이 있으면 이를 피쳐 벡터로 사용하여, 콜드스타트 문제를 완화시킬 수도 있다. 본 논문에서는 순수한 협업필터링 방법에 대하여만 다룬다. 그러므로 이 피쳐벡터들은 원-핫 인코딩이다.

인풋레이어(원핫인코딩)을 지나면 임베딩 레이어가 있다. sparse한 원핫벡터를 fully-connected layer에 통과시켜 밀집 벡터를 얻는다. 그렇게 해서 얻어진 유저 또는 아이템 임베딩은 latent vector라고도 볼 수 있다. 그 다음, 유저 임베딩과 아이템 임베딩은 멀티레이어 신경망으로 들어간다. 우린 이 멀티레이어 신경망을 **neural collaborative filtering layers**라고 부르기로 한다. 이것은 latent vector를 예측 스코어로 매핑한다.

각 레이어는 커스터마이징 될 수 있다. 마지막 아웃풋 레이어는 예측 스코어 $$\hat{y}_{ui}$$이다. 학습은 이 $$\hat{y}_{ui}$$와 $$y_{ui}$$사이의 point-wise loss를 최소화하는 과정이다.

이제 NCF 예측 모델을 수식화 해보자.

$$
\hat{y}_{ui} = f(\mathbf{P}^T \mathbf{v}_u^U , \mathbf{Q}^T \mathbf{v}_i^I | \mathbf{P}, \mathbf{Q}, \Theta_f)
$$

$$\mathbf{P} \in \mathbb{R}^{M \times K}$$와 $$\mathbf{P} \in \mathbb{R}^{N \times K}$$는 각각 유저와 아이템의 latent factor matrix를 나타낸다. 그리고 $$\Theta_f$$는 상호작용 함수 $$f$$의 모델 파라미터를 나타낸다. 함수 $$f$$는 멀티레이어 신경망으로 정의되기 때문에, 다음과 같이 수식화된다.

$$
f(\mathbf{P}^T \mathbf{v}_u^U ,  \mathbf{Q}^T \mathbf{v}_i^I) = \phi_{out}(\phi_X (... \phi_2(\phi_1(\mathbf{P}^T \mathbf{v}_u^U ,  \mathbf{Q}^T \mathbf{v}_i^I))))
$$

$$\phi_{out}$$과 $$\phi_{x}$$는 각각 아웃풋 레이어, $$x$$번째 neural collaborative filtering 레이어를 나타낸다. output 레이어를 제외하면 총 $$X$$개의 neural CF 레이어가 있다.

#### 3.1.1 Learning NCF

모델 파라미터를 학습시키기 위해서, 기존 pointwise 방법론은 squared loss에 대해 회귀를 수행한다.

$$
L_{sqr} = \sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} w_{ui}(y_{ui} - \hat{y}_{ui})^2
$$

$$\mathcal{Y}$$는 $$\mathbf{Y}$$에서 관측된 상호작용의 집합을 의미하고, $$\mathcal{Y}^{-}$$는 네거티브 샘플을 의미하는데 네거티브 전체가 될 수도 있고 샘플링된 것일 수도 있다. 그리고 $$w_{ui}$$는 하이퍼 파라미터로써 학습 데이터 $$(u, i)$$의 가중치이다. squared loss는 가데이터가 가우시안 분포로부터 생성되었다고 가정하는데\[29], 우리는 이것이 implicit data에는 부합하지 않는다고 보았다. 왜냐하면 implicit data에게 타겟 값은 1 또는 0인 바이너리 값이다. 그래서 우리는 pointwise NCF를 학습시키기 위하여 확률적인 접근을 제안한다.

implicit 피드백이 한 가지 클래스라는 점을 고려했을 때 우리는 $$y_{ui}$$를 라벨로 볼 수 있다 -> 1은 아이템 $$i$$가 $$u$$와 상관이 있다, 0은 그렇지 않다로. 예측 스코어 $$\hat{y}_{ui}$$는 $$i$$가 $$u$$와 얼만큼 상관있는지 나타낸다. NCF에게 이러한 확률적 설명력을 부여하기 위해서 아웃풋 $$\hat{y}_{ui}$$의 범위를 $$[0,1]$$로 제한해야한다. 이는 아웃풋 레이어 $$\phi_{out}$$에 확률 함수(예를 들면 Logistic 또는 Probit 함수)를 사용함으로서 쉽게 해결된다. 위 설정으로 우리는 likelihood function을 아래와 같이 정의할 수 있다:

$$
p(\mathcal{Y}, \mathcal{Y}^- | \mathbf{P}, \mathbf{Q}, \Theta_f)=\prod_{(u,i) \in \mathcal{Y}}\hat{y}_{ui} \prod_{(u,j) \in \mathcal{Y}^-}(1-\hat{y}_{uj})
$$

likelihood에 네거티브 로그를 취함으로써, 다음과 같이 쓸 수 있다.

$$
\begin{align*}

L&=-\sum_{(u,i) \in 


\mathcal{Y}}\log \hat{y}_{ui} - \sum_{(u,j) \in \mathcal{Y}^-}\log (1-\hat{y}_{uj})
\\
&= -\sum_{(u,i)\in \mathcal{Y} \cup \mathcal{Y}^-} y_{ui}\log \hat{y}_{ui} + (1 - y_{ui}) \log(1-\hat{y}_{ui})
\end{align*}
$$

**이것이 NCF가 최소화해야하는 목적함수이다!!** 그리고 최적화는 stochastic gradient descent(SGD)를 통해 수행될 수 있다. 눈치 빠른 독자는 이것이 log loss라고도 하는 binary cross-entropy loss와 똑같이 생긴 것을 알아챘을 것이다. 우린 확률적 치료법(?)을 사용하여 implicit 피드백을 사용하는 추천 문제를 binary classification 문제로 다룬다. 추천 논문들 중에 분류문제로 다룬 논문이 거의 없기 때문에 4.3에서 이 문제에 대해 다룰 것이다. 네거티브 샘플 $$\mathcal{Y}^-$$에 대하여, 균일한 분포로 샘플링한다. 균일하지 않은 확률로 샘플링하는 것(예를 들면 아이템 인기도에 따라\[12])이 성능을 더 향상시킬 수 있겠으나, 이는 추후 연구로 남겨놓도록 한다.



### 3.2 Generalized Matrix Factorization (GMF)

이젠 MF가 어떻게 NCF의 특이한 케이스로 해석될 수 있는지 보여주겠다. MF는 추천시스템에서 가장 유명한 모델이고 많이 연구돼왔기 때문에, MF를 다룸으로써 NCF가 범 factorization 모델에 속한다는 것을 알 수 있다\[26].

유저(아이템)의 원 핫 인코딩에 의해 얻어진 임베딩 벡터는 유저(아이템)의 latent vector로 볼 수 있다. 유저 latent vector $$\mathbf{p}_u$$를 $$\mathbf{P}^T\mathbf{v}_u^U$$라고 하고, 아이템 latent vector $$\mathbf{q}_i$$를 $$\mathbf{Q}^T\mathbf{v}_i^I$$라고 하자. 첫 번째 neural CF layer의 매핑 함수를 다음과 같이 정의한다.

$$
\phi_1(\mathbf{p}_u,\mathbf{q}_i)=\mathbf{p}_u \odot \mathbf{q}_i
$$

$$\odot$$은 벡터의 element-wise 곱을 나타낸다. 그 다음 이 벡터를 아웃풋 레이어에 사영시킨다.

$$
\hat{y}_{ui}=a_{out}(\mathbf{h}^T(\mathbf{p}_u \odot \mathbf{q}_i))
$$

$$a_{out}$$과 $$\mathbf{h}$$는 각각 활성함수와 아웃풋 레이어의 가중치를 나타낸다. 직관적으로, 만약 $$a_{out}$$로 identity function을 사용하고 $$\mathbf{h}$$를 1로 이루어진 uniform 벡터로 강제한다면, 이건 정확히 MF모델이다.

NCF 프레임워크에서 MF는 쉽게 일반화, 확장될 수 있다. 예를 들어, 만약 $$\mathbf{h}$$를 uniform constraint가 없이 데이터로부터 학습시킨다면 다양한 latent dimension의 중요도를 판별할 수 있는 MF의 변형이 된다. 그리고 만약 $$a_{out}$$을 비선형 함수로 사용한다면, 이는 MF를 비선형적인 설정까지 일반화시켜서 선형 MF모델보다 더욱 표현력이 강해질 것이다. 본 논문에서는 시그모이드 함수 $$\sigma(x)=1/(1+e^{-x})$$를 $$a_{out}$$로 사용하고, $$\mathbf{h}$$를 log loss(Section 3.1.1)로 학습시킴으로써 NCF 하에서 MF의 일반화 버전을 수행한다.

### 3.3 Multi-Layer Perceptron (MLP)

NCF는 유저와 아이템을 모델링하기 위해 두 가지 길을 채택했기 때문에, 그들을 concatenating함으로써 두 가지 길의 피쳐를 결합하는것이 직관적이다. 이 방법은 멀티모달 딥러닝 연구에서 널리 채택되어왔다\[47, 34]. 그러나, 단순한 벡터 concatenation은 유저와 아이템 latent feature 사이의 상호작용을 고려하지 않는다. 이는 collaborative filtering 효과를 모델링하는데 불충분하다. 이 문제를 해결하기 위해서 concatenated 벡터에 대해 표준적인 MLP의 히든 레이어를 추가하여 유저와 아이템 latent feature 사이의 상호작용을 학습하게 할 것을 제안한다. 이렇게 하면 GMF에서 고정된 element-wise 곱을 한 방식보다 $$\mathbf{p}_u$$와 $$\mathbf{q}_i$$사이의 상호작용을 학습하기 위하여 모델에게 더 많은 유연성과 비선형성을 줄 수 있다. 더욱 정확히는, NCF 프레임워크 하에서의 MLP 모델은 다음과 같이 정의된다:

$$
\begin{align*}
\mathbf{z}_1 &= \phi_1 (\mathbf{p}_u, \mathbf{q}_i) = 
\begin{bmatrix}
\mathbf{p}_u  \\
\mathbf{q}_i
\end{bmatrix},
\\
\phi_2(\mathbf{z}_1) &= a_2(W_2^T \mathbf{z}_1 + \mathbf{b}_2),

\\
& \cdots
\\
\phi_L(\mathbf{z}_{L-1}) &= a_L(W_L^T \mathbf{z}_{L-1} + \mathbf{b}_L)
\\
\hat{y}_{ui} &= \sigma(\mathbf{h}^T \phi _L (\mathbf{z}_{L-1}))


\end{align*}
$$

$$W_x$$, $$b_x$$, $$a_x$$는 각각 $$x$$번째 레이어의 퍼셉트론에 대한 가중치 행렬, 편향 벡터, 활성화함수를 나타낸다. MLP 레이어의 활성화 함수는 시그모이드, 하이퍼볼릭 탄젠트, ReLU, 또 다른 것들 중 자유롭게 선택할 수 있다. 각각의 함수에 대해 설명하자면: 1) 시그모이드 함수는 각각의 뉴런을 (0, 1) 안에 위치시켜서 모델의 성능을 제한할 수도 있다; 이는 또한 saturation이라는 단점이 있는데, 즉 아웃풋이 0 또는 1에 가까워지게 되면 뉴런이 학습을 멈춰버린다는 것이다. 2) tanh는 좋은 선택지이고 널리 선택되어 왔지만 \[6, 44], 이는 오직 시그모이드의 단점만을 어느 정도만 완화시킨다. 왜냐하면 이는 시그모이드가 스케일링된 버전이라고 볼 수 있기 때문이다($$\tanh (x/2)=2\sigma - 1$$). 3)그리고 우리는 ReLU를 선택했는데, 이는 생물학적으로 더 그럴듯하고 non-saturated인 것으로 밝혀졌었다\[9]; 게다가, 이는 sparse 활성화를 촉진시켜서 sparse 데이터에 더 잘 맞고 오버피팅될 확률이 적다. 우리의 실험에서는 ReLU가 tanh보다 살짝 더 나은 결과를 보여주었고 sigmoid보다는 크게 좋았다.

신경망 구조의 디자인에 관해서는, 일반적인 솔루션은 타워 패턴을 따르는 것, 즉 아래 레이어가 가장 넓고 각각의 이어지는 레이어는 더 적은 수의 뉴런을 갖는 구조다(Figure 2). 더 높은 레이어의 히든 유닛이 더 적은 수가 됨으로써 데이터의 피쳐의 더욱 추상적인 특징을 학습할 수 있다는 것이 전제이다\[10]. 우리는 실험에서 높은 층으로 갈수록 레이어 사이즈를 절반으로 줄이는 타워 구조를 사용했다.



### 3.4 Fusion of GMF and MLP















































\






