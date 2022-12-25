# NeuMF(Neural Collaborative Filtering)

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

<figure><img src="../.gitbook/assets/image (12).png" alt=""><figcaption></figcaption></figure>

파라미터 $$\Theta$$를 추정하기 위해서, 기존의 접근법들은 일반적으로 목적함수를 최적화하는 머신러닝 패러다임을 따랐다. 목적함수의 두 가지 유형이 본 논문에서 가장 많이 다뤄질 것이다 - pointwise loss \[14, 19], pairwise loss \[27, 33]. 명시적 피드백에 대한 많은 연구들의 연장선으로써, pointwise learning 방법은 보통 $$\hat{y}_{ui}$$와 타겟 값 $$y_{ui}$$ 사이의 squared loss를 최소화하는 회귀 방식을 따른다. 네거티브 데이터를 안쓰는 대신, 관측되지 않은 모든 엔트리 또는 샘플링된 인스턴스를 네거티브 피드백으로 사용한다 \[14]. pairwise learning \[27, 44]의 아이디어는 관측된 엔트리가 관측되지 않은 것보다 높게 랭크되어야 한다는 것이다. 그러게 하면, $$\hat{y}_{ui}$$와 $$y_{ui}$$ 사이의 손실을 최소화하는 대신, 관측된 엔트리 $$\hat{y}_{ui}$$와 관측되지 않은 엔트리 $$y_{ui}$$사이의 margin, 즉 간격을 최대화한다.

NCF는 $$\hat{y}_{ui}$$를 예측하기 위하여, 신경망을 이용하여 상호작용 함수 $$f$$를 파라미터화 한다. 이렇게 하면 pointwise와 pairwise를 모두 사용할 수 있다.

### 2.2 Matrix Factorization <a href="#31-general-framework" id="31-general-framework"></a>

데이터사이언스 기본 - Matrix Factorization으로 이동



## 3. NEURAL COLLABORATIVE FILTERING

### 3.1 General Framework

<figure><img src="../.gitbook/assets/image (10).png" alt=""><figcaption><p>Figure 2</p></figcaption></figure>

유저 $$u$$와 아이템 $$i$$의 상호작용을 $$y_{ui}$$라고 하자. Figure 2에서 가장 아래 쪽에 있는 인풋 레이어는 두 개의 피쳐 벡터들로 구성되어있다. 이를 각각 유저 $$u$$와 아이템 $$i$$를 가리키는 피쳐벡터, $$\mathbf{v}_u^U$$와 $$\mathbf{v}_i^I$$라고 하자. 이 피쳐벡터들은 경우에 따라 커스터마이징이 가능하다. 즉, 유저나 아이템의 side-information이 있으면 이를 피쳐 벡터로 사용하여, 콜드스타트 문제를 완화시킬 수도 있다. 본 논문에서는 순수한 협업필터링 방법에 대하여만 다룬다. 그러므로 이 피쳐벡터들은 원-핫 인코딩이다.

인풋레이어(원핫인코딩)을 지나면 임베딩 레이어가 있다. sparse한 원핫벡터를 fully-connected layer에 통과시켜 밀집 벡터를 얻는다. 그렇게 해서 얻어진 유저 또는 아이템 임베딩은 latent vector라고도 볼 수 있다. 그 다음, 유저 임베딩과 아이템 임베딩은 멀티레이어 신경망으로 들어간다. 우린 이 멀티레이어 신경망을 **neural collaborative filtering layers**라고 부르기로 한다. 이것은 latent vector를 예측 스코어로 매핑한다.

각 레이어는 커스터마이징 될 수 있다. 마지막 아웃풋 레이어는 예측 스코어 $$\hat{y}_{ui}$$이다. 학습은 이 $$\hat{y}_{ui}$$와 $$y_{ui}$$사이의 point-wise loss를 최소화하는 과정이다.

이제 NCF 예측 모델을 수식화 해보자.

$$
\hat{y}_{ui} = f(\mathbf{P}^T \mathbf{v}_u^U , \mathbf{Q}^T \mathbf{v}_i^I | \mathbf{P}, \mathbf{Q}, \Theta_f)
$$

$$\mathbf{P} \in \mathbb{R}^{M \times K}$$와 $$\mathbf{Q} \in \mathbb{R}^{N \times K}$$는 각각 유저와 아이템의 latent factor matrix를 나타낸다. 그리고 $$\Theta_f$$는 상호작용 함수 $$f$$의 모델 파라미터를 나타낸다. 함수 $$f$$는 멀티레이어 신경망으로 정의되기 때문에, 다음과 같이 수식화된다.

$$
f(\mathbf{P}^T \mathbf{v}_u^U ,  \mathbf{Q}^T \mathbf{v}_i^I) = \phi_{out}(\phi_X (... \phi_2(\phi_1(\mathbf{P}^T \mathbf{v}_u^U ,  \mathbf{Q}^T \mathbf{v}_i^I))))
$$

$$\phi_{out}$$과 $$\phi_{x}$$는 각각 아웃풋 레이어, $$x$$번째 neural collaborative filtering 레이어를 나타낸다. output 레이어를 제외하면 총 $$X$$개의 neural CF 레이어가 있다.

#### 3.1.1 Learning NCF

기존의 pointwise 방법론\[14, 39]은 모델 파라미터를 학습시키기 위해 squared loss에 대한 회귀를 수행했다.

$$
L_{sqr} = \sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} w_{ui}(y_{ui} - \hat{y}_{ui})^2
$$

$$\mathcal{Y}$$는 $$\mathbf{Y}$$에서 관측된 상호작용의 집합을 의미하고, $$\mathcal{Y}^{-}$$는 네거티브 샘플을 의미하는데 네거티브 전체가 될 수도 있고 샘플링된 것일 수도 있다. 그리고 $$w_{ui}$$는 하이퍼 파라미터로써 학습 데이터 $$(u, i)$$의 가중치이다. squared loss는 가데이터가 가우시안 분포로부터 생성되었다고 가정하는데\[29], **우리는 이것이 implicit data에는 부합하지 않는다고 보았다.** 왜냐하면 implicit data에서 타겟 값 $$y_{ui}$$는 $$u$$가 $$i$$와 상호작용 했는지 안했는지를 나타내는 1 또는 0인 바이너리 값이기 때문이다. 그래서 우리는 pointwise NCF를 학습시키기 위하여 확률적인 접근을 제안한다.

implicit 피드백이 한 가지 클래스라는 점을 고려했을 때 **우리는** $$y_{ui}$$**를 라벨로 볼 수 있다** --> 1은 아이템 $$i$$가 $$u$$와 상관이 있다는 1, 그렇지 않다는 0으로. 예측 스코어 $$\hat{y}_{ui}$$는 $$i$$가 $$u$$와 얼만큼 상관있는지 나타낸다. NCF에게 이러한 확률적 설명력을 부여하기 위해서 아웃풋 $$\hat{y}_{ui}$$의 범위를 $$[0,1]$$로 제한해야한다. 이는 아웃풋 레이어 $$\phi_{out}$$에 확률 함수(예를 들면 Logistic 또는 Probit 함수)를 사용함으로서 쉽게 해결된다. 위와 같은 설정으로 우리는 likelihood function을 아래와 같이 정의할 수 있다:

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

**이것이 NCF가 최소화 해야하는 목적함수이다!!** 그리고 최적화는 stochastic gradient descent(SGD)를 통해 수행될 수 있다. 눈치 빠른 독자는 이것이 log loss라고도 하는 binary cross-entropy loss와 똑같이 생긴 것을 알아챘을 것이다. 우린 확률적 치료법(?)을 사용하여 implicit 피드백을 사용하는 추천 문제를 binary classification 문제로 다룬다. 추천 논문들 중에 분류문제로 다룬 논문이 거의 없기 때문에 4.3에서 이 문제에 대해 다룰 것이다. 네거티브 샘플 $$\mathcal{Y}^-$$에 대하여, 균일한 분포로 샘플링한다. 균일하지 않은 확률로 샘플링하는 것(예를 들면 아이템 인기도에 따라\[12])이 성능을 더 향상시킬 수 있겠으나, 이는 추후 연구로 남겨놓도록 한다.



### 3.2 Generalized Matrix Factorization (GMF)

이젠 MF가 어떻게 NCF의 특이한 케이스로 해석될 수 있는지 보여주겠다. MF는 추천시스템에서 가장 유명한 모델이고 많이 연구돼왔기 때문에, MF를 다룸으로써 NCF가 범 factorization 모델에 속한다는 것을 알 수 있다\[26].

유저(아이템)의 원-핫 인코딩에 의해 얻어진 임베딩 벡터는 유저(아이템)의 latent vector로 볼 수 있다. 유저 latent vector $$\mathbf{p}_u$$를 $$\mathbf{P}^T\mathbf{v}_u^U$$라고 하고, 아이템 latent vector $$\mathbf{q}_i$$를 $$\mathbf{Q}^T\mathbf{v}_i^I$$라고 하자. 첫 번째 neural CF layer의 매핑 함수를 다음과 같이 정의한다.

$$
\phi_1(\mathbf{p}_u,\mathbf{q}_i)=\mathbf{p}_u \odot \mathbf{q}_i
$$

$$\odot$$은 벡터의 element-wise 곱을 나타낸다. 그 다음 이 벡터를 아웃풋 레이어에 사영시킨다.

$$
\hat{y}_{ui}=a_{out}(\mathbf{h}^T(\mathbf{p}_u \odot \mathbf{q}_i))
$$

<figure><img src="../.gitbook/assets/image (8) (1).png" alt=""><figcaption></figcaption></figure>

$$a_{out}$$과 $$\mathbf{h}$$는 각각 활성함수와 아웃풋 레이어의 가중치를 나타낸다. 직관적으로, 만약 $$a_{out}$$로 identity function을 사용하고 $$\mathbf{h}$$를 1로 이루어진 uniform 벡터로 강제한다면, 이건 정확히 MF모델이다.

NCF 프레임워크 안에서 MF는 쉽게 일반화, 확장될 수 있다. 예를 들어, 만약 $$\mathbf{h}$$를 uniform constraint 없이 데이터로부터 학습시킨다면 다양한 latent dimension의 중요도를 판별할 수 있는 MF의 변형이 된다. 그리고 만약 $$a_{out}$$을 비선형 함수로 사용한다면, 이는 MF를 비선형적인 설정까지 일반화시켜서 선형 MF모델보다 더욱 표현력이 강해질 것이다. 본 논문에서는 시그모이드 함수 $$\sigma(x)=1/(1+e^{-x})$$를 $$a_{out}$$로 사용하고, $$\mathbf{h}$$를 log loss(Section 3.1.1)로 학습시킴으로써 NCF 하에서 MF의 일반화 버전을 수행한다.

### 3.3 Multi-Layer Perceptron (MLP)

NCF는 유저와 아이템을 모델링하기 위해 두 가지 길을 채택했기 때문에, 그들을 concatenating함으로써 두 가지 길의 피쳐를 결합하는것이 직관적이다. 이 방법은 멀티모달 딥러닝 연구에서 널리 채택되어왔다\[47, 34]. 그러나, 단순한 벡터 concatenation은 유저와 아이템 latent feature 사이의 상호작용을 고려하지 않는다. 이는 collaborative filtering 효과를 모델링하는데 불충분하다. **이 문제를 해결하기 위해서 concatenated 벡터에 대해 표준적인 MLP의 히든 레이어를 추가하여 유저와 아이템 latent feature 사이의 상호작용을 학습하게 할 것을 제안한다.** 이렇게 하면 GMF에서 고정된 element-wise 곱을 한 방식보다 $$\mathbf{p}_u$$와 $$\mathbf{q}_i$$사이의 상호작용을 학습하기 위하여 모델에게 더 많은 유연성과 비선형성을 줄 수 있다. 더욱 정확히는, NCF 프레임워크 하에서의 MLP 모델은 다음과 같이 정의된다:

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

<figure><img src="../.gitbook/assets/image (2) (2).png" alt=""><figcaption></figcaption></figure>

$$W_x$$, $$b_x$$, $$a_x$$는 각각 $$x$$번째 레이어의 퍼셉트론에 대한 가중치 행렬, 편향 벡터, 활성화함수를 나타낸다. MLP 레이어의 활성화 함수는 시그모이드, 하이퍼볼릭 탄젠트, ReLU, 또 다른 것들 중 자유롭게 선택할 수 있다. 각각의 함수에 대해 설명하자면: 1) 시그모이드 함수는 각각의 뉴런을 (0, 1) 안에 위치시켜서 모델의 성능을 제한할 수도 있다; 이는 또한 saturation이라는 단점이 있는데, 즉 아웃풋이 0 또는 1에 가까워지게 되면 뉴런이 학습을 멈춰버린다는 것이다. 2) tanh는 좋은 선택지이고 널리 선택되어 왔지만 \[6, 44], 이는 오직 시그모이드의 단점만을 어느 정도만 완화시킨다. 왜냐하면 이는 시그모이드가 스케일링된 버전이라고 볼 수 있기 때문이다($$\tanh (x/2)=2\sigma - 1$$). 3)그리고 우리는 ReLU를 선택했는데, 이는 생물학적으로 더 그럴듯하고 non-saturated인 것으로 밝혀졌었다\[9]; 게다가, 이는 sparse 활성화를 촉진시켜서 sparse 데이터에 더 잘 맞고 오버피팅될 확률이 적다. 우리의 실험에서는 **ReLU**가 tanh보다 살짝 더 나은 결과를 보여주었고 sigmoid보다는 크게 좋았다.

신경망 구조의 디자인에 관해서는, 일반적인 솔루션은 타워 패턴을 따르는 것, 즉 아래 레이어가 가장 넓고 각각의 이어지는 레이어는 더 적은 수의 뉴런을 갖는 구조다(Figure 2). 더 높은 레이어의 히든 유닛이 더 적은 수가 됨으로써 데이터의 피쳐의 더욱 추상적인 특징을 학습할 수 있다는 것이 전제이다\[10]. 우리는 높은 층으로 갈수록 layer의 사이즈를 절반으로 줄이는 타워 구조를 사용했다.



### 3.4 Fusion of GMF and MLP

지금까지 우리는 NCF의 두 가지 예시를 개발했다. latent feature 상호작용을 모델링하기 위하여 선형 커널을 적용한 GMF, 데이터로부터 상호작용 함수를 학습하기 위하여 비선형 커널을 사용하는 MLP. 여기서 질문이 하나 생긴다. 복잡한 유저-아이템 상호작용을 더 잘 모델링하기 위하여, GMF와 MLP가 상호 보완적으로 **NCF 프레임워크 안에서 어떻게 GMF와 MLP를 결합시켜야 할까**?

바로 적용할 수 있는 방법은 GMF와 MLP가 같은 임베딩 레이어를 공유하게 만들어서, 그 이후 그들의 상호작용 함수의 아웃풋을 결합하는 것이다. 이 방법은 잘 알려진 Neural Tensor Network(NTN) \[33]과 같은 정신을 공유한다. 구체적으로, GMF를 한 개의 레이어를 가진 MLP와 결합하는 것은 다음과 같이 수식화된다:

$$
\hat{y}_{ui} = \sigma(\mathbf{h}^T a(\mathbf{p}_u \odot \mathbf{q}_i) + \mathbf{W} 
\begin{bmatrix}
\mathbf{p}_u  \\
\mathbf{q}_i
\end{bmatrix}+\mathbf{b}

)
$$

<figure><img src="../.gitbook/assets/image (5) (2).png" alt=""><figcaption></figcaption></figure>

그러나, GMF와 MLP의 임베딩을 공유하는 것은 성능을 제한할 수도 있다. 예를 들어, GMF와 MLP는 같은 임베딩 사이즈를 사용해야만 한다; 최적의 임베딩 사이즈가 매우 다른 데이터라면 이 방법은 최적의 앙상블이라고 할 수 없다.

결합 모델의 유연성을 확보하기 위하여, **GMF와 MLP가 각각의 임베딩을 학습하도록 한 후 마지막 히든레이어에서 concatenating하도록 했다.** Figure3이 이 제안을 묘사하고 있으며, 수식은 다음과 같다:

$$
\begin{align*}
\phi^{GMF} &= \mathbf{p}_u^G \odot \mathbf{q}_i^G,
\\
\phi^{MLP} &= a_L(\mathbf{W}_L^T (a_{L-1}(\dots a_2(\mathbf{W}_2^T \begin{bmatrix}
\mathbf{p}_u^M  \\
\mathbf{q}_i^M
\end{bmatrix} + \mathbf{b}_2 ) \dots )) + \mathbf{b}_L
\\
\hat{y}_{ui} &= \sigma(\mathbf{h}^T \begin{bmatrix}
\phi^{GMF}  \\
\phi^{MLP}
\end{bmatrix})
\end{align*}
$$

<figure><img src="../.gitbook/assets/image (2) (3).png" alt=""><figcaption><p>Figure 3. Neural matrix factorization model</p></figcaption></figure>

$$\mathbf{p}_u^G$$와 $$\mathbf{p}_u^M$$은 GMF와 MLP의 유저 임베딩을 나타낸다. $$\mathbf{q}_i^G$$와 $$\mathbf{q}_i^M$$는 아이템에 대하여 마찬가지다. 위에서 언급했듯, MLP의 활성함수로 ReLU를 사용했다. 이 모델은 MF의 선형성과 DNN의 비선형성을 결합하여 유저-아이템 latent 구조를 모델링한다. 우리는 이 모델을 "_NeuMF(Neural Matrix Factorization_)"이라고 부른다. 각각의 모델 파라미터들에 관한 미분값은 표준적인 역전파로 계산될 수 있다.

{% code overflow="wrap" lineNumbers="true" %}
```python
class NeuMF(nn.Module):
    def __init__(self, n_user, n_item, emb_dim):
        super().__init__()
        # GMF
        self.GMF_user = nn.Embedding(n_user, emb_dim)
        self.GMF_item = nn.Embedding(n_item, emb_dim)

        # MLP
        self.MLP_user = nn.Embedding(n_user, emb_dim)
        self.MLP_item = nn.Embedding(n_item, emb_dim)
        self.MLP_linear = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, emb_dim//(2*2)),
            nn.ReLU(),
            nn.Linear(emb_dim//(2*2), emb_dim//(2*2*2)),
        )

        # output layer
        self.output_layer = nn.Linear(emb_dim + (emb_dim//(2*2*2)), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        # GMF
        phi_gmf = torch.mul(self.GMF_user(user), self.GMF_item(item))

        # MLP
        concat = torch.cat([self.MLP_user(user), self.MLP_item(item)], dim=1)
        phi_mlp = self.MLP_linear(concat)
        
        # output layer
        output = self.output_layer(torch.cat([phi_gmf, phi_mlp], dim=1))
        logit = self.sigmoid(output)

        return logit
```
{% endcode %}



#### 3.4.1 Pre-training

NeuMF의 목적함수가 non-convexity를 갖고 있기 때문에, 그래디언트 기반의 최적화 방법은 오직 로컬 옵티멈만을 찾는다. \[7]에서 초기화가 수렴과 딥러닝 모델의 성능에 있어서 중요하다는 것이 보여졌다. NeuMF는 GMF와 MLP의 앙상블이기 때문에, GMF와 MLP의 사전학습 모델을 이용하여 NeuMF를 초기화 할 것을 제안한다.

우선 GMF와 MLP를 수렴까지 랜덤 초기화한다. 그리고 난 후 그 모델 파라미터를 NeuMF의 파라미터의 상응하는 부분에 대해 초기화 하는데 사용한다. 변동성은 오직 아웃풋 레이어에 있는데, 이 부분은 두 모델의 가중치를 다음과 같이 concatenate하여,

$$
\mathbf{h} \leftarrow
\begin{bmatrix}
\alpha \mathbf{h}^{GMF}  \\
(1-\alpha)\mathbf{h}^{MLP}
\end{bmatrix}
$$

$$\mathbf{h}^{GMF}$$와 $$\mathbf{h}^{MLP}$$는 사전학습된 GMF와 MLP 모델의 $$\mathbf{h}$$ 벡터이다. $$\alpha$$는 두 사전학습 모델 사이의 트레이드 오프를 결정하는 하이퍼 파라미터이다.

GMF와 MLP의 학습에 관해 말하자면, 최적화 기법으로 Adam(Adaptive Moment Estimation)\[20]을 사용하였다. Adam은 두 모델에 대하여 vanilla SGD보다 빨리 수렴하고 learning rate 조정이 필요없다. NeuMF에 사전학습 파라미터를 삽입한 뒤에는 Adam이 아닌 vanilla SGD를 사용하여 최적화하였다. 왜냐하면 Adam은 파라미터 업데이트 시 모멘텀 정보를 저장해야하기 때문이다. NeuMF를 사전학습된 모델 파라미터로 초기화하고 모멘텀 정보는 버리기 때문에, NeuMF를 모멘텀 기반의 방법으로 더욱 최적화 시키는것은 적절하지 않다.



## 4. EXPERIMENTS

이 섹션에서는 다음 research question에 대답하기 위하여 실험을 수행한다.

**RQ1.** 우리가 제안하는 NCF 방법이 SOTA의 implicit collaborative filtering을 능가하는가?

**RQ2.** 제안한 최적화 프레임워크(log loss with negative sampling)가 추천 태스크에서는 어떻게 동작하는가?

**RQ3.** 더 깊은 히든레이어층이 유저-아이템 상호작용 데이터로부터 학습에 대해 유용한가?

아래에서는 실험 세팅 및 위 research question에 대한 대답을 한다.

### 4.1 Experimental Settings

**Datasets.** 두 가지의 공용 데이터셋 MovieLens와 Pinterest로 실험하였다. 두 데이터셋 요약은 Table 1과 같다.

<figure><img src="../.gitbook/assets/image (1) (1) (2).png" alt=""><figcaption></figcaption></figure>

1. MovieLens. 이 영화 평점 데이터셋은 collaborative filtering 알고리즘을 계산하기 위하여 널리 이용되어 왔다. 백만 개의 평점이 있는 버전을 사용하였고, 여기서 각 유저는 최소 20개 평가를 하였다. 이건 explicit 피드백 데이터이기 때문에, 우린 의도적으로 explicit 피드백의 implicit signal\[21] 로부터의 학습 성능을 조사하기 위해 이 데이터셋을 선택했다. 결국, implicit data로 바꾸어 평가를 했는지 안했는지에 대해 1과 0으로 나타냈다는 것이다.
2. Pinterest. 이 implicit 피드백 데이터는 \[8]에 의해 컨텐츠 기반 추천을 위하여 구축되었다. 오리지널 데이터는 매우 크고 매우 sparse하다. 예를 들어, 20% 이상의 유저가 오직 하나의 pin을 갖고 있어서 collaborative filtering 알고리즘을 계산하기 어렵다. 그래서 movielens 데이터와 같은 방식으로 필터링하여 최소 20번의 상호작용(pin)이 있는 유저만을 사용했다. 이렇게 하면 55,187명의 유저와 1,500,809번의 상호작용이 존재한다. 이 데이터에서의 상호작용이란 유저가 자신의 보드에 이미지를 pin 한 것을 의미한다.

**Evaluation Protocols.** 아이템 추천 성능을 계산하기 위하여, \[1, 14, 27]과 같은 논문에서 널리 사용되는 leave-one-out 평가 방법을 채택하였다. 각 유저에 대하여 가장 최신 상호작용을 테스트셋으로 정하고 나머지 데이터를 학습에 사용한다. 모든 아이템을 각 유저에 대해서 랭킹하는 것은 매우 시간낭비이기 때문에, 상호작용 없는 아이템 중 랜덤 샘플 100개를 뽑아 테스트 아이템을 100개 중에서 랭킹하는 방식\[6, 21]을 사용했다. 랭킹된 리스트의 성능은 _Hit Ratio_(HR)와 nDCG\[11]로 측정된다. 특별한 언급이 없다면 랭킹 리스트를 10개에서 끊어서, HR은 테스트 아이템이 그 top10에 존재하는지를 측정하고, NDCG는 그 hit의 위치까지 고려하여, 높이 랭크될수록 높은 스코어를 할당한다. 각 테스트 유저에 대해 두 가지 메트릭을 모두 사용하여 평균값을 적어놓았다.



**Baselines.** NCF(GMF, MLP, NeuMF)를 다음의 방법들과 비교하였다.

* **ItemPop.** 아이템들이 상호작용의 수에 따른 인기도에 의해 랭크된다. 이는 개인화 추천 방법은 아니다\[27].
* **ItemKNN \[31].** 이는 표준적인 아이템 기반의 협업필터링 방법이다. implicit data에 맞도록 적용하기 위해 \[19]의 세팅을 따랐다.
* **BPR \[27].** 이 방법은 implicit 피드백을 학습할 수 있도록 맞춰진 pairwise ranking loss로 수식2의 MF 모델을 최적화한다. 이는 아이템 추천의 베이스라인으로서 매우 경쟁력이 있는 방법이다. 우리는 고정된 learning rate를 사용했으며, 다양하게 시도하여 최고 성능을 적어놓았다.

$$
\hat{y}_{ui}=f(u,i|\mathbf{p}_u, \mathbf{q}_i)=\mathbf{p}_u^T\mathbf{q}_i=\sum_{k=1}^{K}p_{uk}q_{ik}
$$

* **eALS \[14].** 아이템 추천에 있어서 SOTA MF 방법이다. 이는 수식 5의 squared loss를 최적화하며, 아이템 인기도에 따라 비균일하게 가중치를 적용하여, 사용하지 않은 모든 아이템들을 네거티브 샘플로 취급한다. eALS는 균일하게 가중치를 주는 WMF \[19]보다 좋은 성능을 보이며, 여기서는 WMF의 성능은 다루지 않는다.

$$
L_{sqr} = \sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} w_{ui}(y_{ui} - \hat{y}_{ui})^2
$$

**Parameter Settings.** Keras로 실험을 수행했다. NCF의 하이퍼 파라미터를 결정하기 위하여, 각 유저별로 상호작용 중 임의로 하나를 뽑아 검증데이터로 사용하였고 이에 기반에 하이퍼 파라미터를 결정하였다. 모든 NCF 모델들은 수식 7의 log loss를 최적화함으로써 학습되며, 하나의 positive 샘플 당 4개의 negative 샘플을 사용했다. 처음부터 학습되는 NCF모델에는 가우시안 분포(평균 0, 표준편차 0.01)로 모델 파라미터를 초기화 하였으며 mini-batch Adam\[20] 으로 최적화하였다. \[128, 256, 512, 1024]의 배치사이즈를, \[0.0001, 0.0005, 0.001, 0.005]의 learning rate를 테스트하였다. NCF의 마지막 히든레이어가 모델 capacity를 결정하기 때문에, 이를 _predictive factor_라고 부르기로 하고 \[8, 16, 32, 64] factor를 평가하였다. 큰 factor는 오버피팅을 유발하고 성능을 낮아지게 한다는 점을 유념해야 한다. 특별한 언급이 없다면, MLP의 히든레이어는 3개를 사용했다. 예를 들어, 만약 predictive factor의 사이즈가 8이라면, neural CF layer의 구조는 32 -> 16 -> 8 이 되며, 임베딩 사이즈는 16이다. 사전학습된 NeuMF의 $$\alpha$$는 0.5로 설정하여, 사전 훈련된 GMF와 MLP가 NeuMF의 초기화에 동등하게 기여하게 했다.

### 4.2 Performance Comparison (RQ1)

<figure><img src="../.gitbook/assets/image (8) (2).png" alt=""><figcaption><p>Figure 4</p></figcaption></figure>

Figure 4는 predictive factor의 수에 대한 HR@10과 NDCG@10 결과이다. MF방법인 BPR과 eALS의 predictive factor는 잠재 인자의 수와 동일하다. ItemKNN에서 이웃의 수를 다양하게 실험해본 후 최고의 성능 값을 기록하였다. ItemPop은 성능이 너무 낮아 Figure 4에서는 제외되었다.

첫 번째로, NeuMF가 두 데이터셋 모두에서 eALS와 BPR과는 큰 차이로 제일 높은 성능을 달성한 것을 볼 수 있다(차이 각각 4.5%, 4.9%). Pinterest에서는 예측인자가 8로 작음에도 불구하고 eALS와 BPR이 일 때보다 높다. 이는 선형적인 MF와 비선형적인 MLP의 결합으로 인해 NeuMF의 표현력이 높아짐을 의미한다. 두 번째로, 서로 다른 두 NCF인 GMF와 MLPr가 모두 좋은 성능을 보여주었다. MLP가 GMF보다 살짝 높다. MLP는 더 많은 히든레이어를 추가함으로써 더 향상될 수 있으며 여기서는 3개일 때만 보여준다. 작은 예측 요인에 대해서는, GMF가 eALS를 두 데이터셋 모두에서 능가했다; 비록 GMF가 큰 인자에서는 오버피팅이 발생하지만, GMF의 최고 성능은 eALS의 최고성능보다 높다. 마지막으로, GMF와 BPR은 같은 MF를 학습하지만 목적함수는 다르기 때문에, 추천태스크에 대한 분류관점의 효율성은 인정하면서, GMF는 BPR보다는 항상 높다.

<figure><img src="../.gitbook/assets/image (5) (3).png" alt=""><figcaption><p>Figure 5</p></figcaption></figure>

Figure 5는 $$K$$를 1부터 10까지 바꿔가며 측정한 Top-$$K$$ 성능을 보여준다. 그림을 깔끔하게 보기 위하여 다른 NCF 모델은 빼로 NeuMF만 기록하였다. NeuMF는 모든 $$K$$에 대해서 항상 다른 모델보다 높다. 그리고 추가적으로 one-sample paired t-test를 수행하여 향상 정도가 $$p<0.01$$로 통계적으로 유의미한지 확인했다. 베이스라인 방법에 대하여, Movielens의 NDCG에서, eALS는 BPR을 5.1% 능가했고 Pinterest에서는 더 낮았다. Pairwise ranking-aware 학습자 때문에 BPR이 랭킹 성능에서는 더 강력하다는 \[14]의 발견과 동일한 결과이다. ItemKNN은 모델 기반 방법론보다는 낮았다. 그리고 ItemPop의 결과는 매우 안좋아서 단순 인기상품보다는 개인화가 요구된다는 것을 보여준다.

#### 4.2.1 Utility of Pre-trainig

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption><p>Table 2. Performance of NeuMF with and without pre-training</p></figcaption></figure>

사전학습의 유용성에 대해 설명하기 위하여, NeuMF의 사전학습 버전과 사전학습 안 된 버전 두 개를 비교했다. 사전학습이 없는 NeuMF에서 Adam과 랜덤 초기화를 사용하여 학습시켰다. Table 2에서 볼 수 있는 것처럼, 사전학습된 NeuMF는 거의 모든 경우 더 좋은 성능을 보였고 오직 MovieLens의 잠재요인 8일 때만 살짝 낮았다. 사전학습된 NeuMF는 Movielens와 Pinterest에서 각각 2,2%, 1,1%만큼 상대적인 향상을 보였다.

### 4.3 Log Loss with Negative Sampling (RQ2)

<figure><img src="../.gitbook/assets/image (2).png" alt=""><figcaption><p>Figure 6. Training loss and recommendation performance of NCF methods w.r.t the number of iterations on MovieLens (factors=8)</p></figcaption></figure>

Implicit 피드백이 한 가지 클래스만 존재한다는 속성을 다루기 위해서, 우리는 추천 문제를 이진 분류 문제로 변환했다. NCF를 확률적 모델로 바라봄으로써, 이를 log loss로 최적화하였다. Figure 6은 MovieLens에 대해 각각의 iteration에서 NCF의 학습 loss와 추천 성능을 보여준다. Pinterest에 대한 결과에서는 동일한 추세를 보여 생략했다. 첫 번째로, iteration이 높아질수록 NCF의 학습 손실은 점점 감소하고 추천 성능은 향상하는 것을 볼 수 있다. iteration 10 이하에서 가장 효율적인 성능 향상이 일어나며, 더 높은 곳에서는 오버피팅이 발생한다(예를 들어, 10 이상에서는 비록 NeuMF의 학습 손실은 감소하지만 추천 성능은 하락한다). 두 번째로, 세 NCF 모델 중 NeuMF가 가장 낮은 학습 손실을 보였고 그 다음 MLP, GMF 순서이다. 추천 성능 또한 NeuMF > MLP > GMF 순서이다. 위 결과는 implicit 피드백을 학습할 때 log loss를 최적화하는 것에 대한 합리성과 효과성을 제공하는 경험적 증거가 된다.

Pointwise 목적 함수\[27 33]에 대한 pointwise log loss의 장점은 네거티브 샘플링 시의 샘플링 비율의 유연성이다. pairwise objective 함수는 오직 하나의 네거티브 샘플이 positive 샘플과 페어링 되는 반면, pointwise loss의 샘플링 비율은 유연하게 조절될 수 있다.

<figure><img src="../.gitbook/assets/image (3) (2).png" alt=""><figcaption><p>Figure 7. positive 샘플당 네거티브 샘플의 수에 따를 성능(factor=16). BPR는 오직 하나의 네거티브 샘플을 사용한다.</p></figcaption></figure>

NCF의 네거티브 샘플링의 효과를 설명하기 위하여, Figure 7에 네거티브 샘플링 비율을 다르게 하여 성능을 비교하였다. positive 샘플 하나 당 하나의 네거티브 샘플링은 최적의 성능을 위해선 충분하지 못하고 더 많은 네거티브 샘플이 좋음이 명백히 보인다. GMF와 BPR을 비교했을 때, GMF의 성능은 네거티브 샘플링 비율이 1일 때는 BPR의 성능과 동일하지만, GMF의 성능은 샘플링 비율이 높아짐에 따라 점점 높아지는 것을 볼 수 있다. 이를 통해 pairwise BPR loss에 대한 log loss의 장점을 확인할 수 있다. 두 모든 데이터셋에 대하여, 최적의 샘플링 비율은 3에서 6 근방이다. Pinterest에서는, 샘플링 비율이 7보다 커지면 성능이 떨어지기 시작했다. 이는 샘플링 비율을 공격적으로 설정하는 것이 성능에 큰 타격이 됨을 보여준다.

### 4.4 Is Deep Learning Helpful? (RQ3)

<figure><img src="../.gitbook/assets/image (5) (4).png" alt=""><figcaption></figcaption></figure>

유저-아이템 상호작용 함수를 신경망으로 학습시키는 연구가 소수 존재함에 따라, 추천 태스크에 딥 네트워크 구조를 사용할지 말지가 궁금해진다. 그러기 위하여, MLP의 히든 레이어의 숫자를 바꿔가며 실험해보았다. 결과는 Table 3과 4에 요약되어있다. MLP-3은 (임베딩 레이어 옆)세 개의 히든 레이어를 가진 MLP를 의미한다. 표에서 볼 수 있는 것처럼, 같은 capability를 가진 모델이라고 해도 더 많은 레이어를 쌓는 것이 더 좋은 성능을 보인다. 이 결과는 collaborative 추천에 대해 딥 모델을 사용하는 것의 효과성을 보여준다. 우리는 이 향상이 더 많은 비선형 레이어를 쌓았기 때문이라고 생각한다. 이를 확인하기 위해서, 활성함수를 idendity 함수로 하여 선형 레이어를 더 쌓아보았다. 그 결과는 ReLU를 사용했을때보다 훨씬 낮았다.

히든레이어가 없는 MLP-0(즉, 임베딩레이어가 곧바로 예측값으로 사영됨)에서는 성능이 매우 낮고 개인화 모델이 아닌 ItemPop보다 좋지도 않았다. 이것은 Section 3.3에서 우리가 **유저와 아이템의 임베딩 벡터를 단순히 concatenating하는 것이 그들의 피쳐 상호작용을 모델링하는데 불충분하고, 히든레이어를 통한 transforming이 필요하다는 주장을 뒷받침한다.**

















\






