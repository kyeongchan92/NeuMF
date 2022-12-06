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

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

Figure1은 내적이 어떻게 MF의 표현력을 제안할 수 있는지를 보여준다. 이를 이해하기 위해 우선적으로 두 가지만 얘기하고 넘어가자. 첫 번째, MF는 유저와 아이템을 같은 latent space로 매핑하기 때문에 두 유저간의 유사도 또한 내적으로 계산될 수 있다(같은 방법으로는 latent vector 사이의 코사인 각도). 두 번째로, 일반화 손실 없이, 우리는 자카드 계수를 MF가 회복시켜야 하는 두 유저간의 ground-truth 유사도로서 사용한다.

Figure 1a의 상위 3개의 row를 보자. $$s_{23}(0.66) > s_{12}(0.5) > s_{13}(0.4)$$를 계산하기는 쉽다. 그렇게 해서, latent space에서의 $$\mathbf{p}_1$$과 $$\mathbf{p}_2$$, 그리고 $$\mathbf{p}_3$$의 기하학적 관계는 Figure 1b처럼 그릴 수 있다. 자 이제 새로운 유저 $$u_4$$를 고려해보자. $$s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)$$를 계산할 수 있는데, 이는 $$u_4$$가 $$u_1$$과 가장 유사하고, 그 다음으로 $$u_3$$와, 그 다음으로 $$u_2$$ 순으로 유사하다는 뜻이다. 그러나, 만약 MF 모델이 $$\mathbf{p}_4$$를 $$\mathbf{p}_1$$에 가장 가깝게 놓는다면 $$\mathbf{p}_4$$가 $$\mathbf{p}_3$$보다 $$\mathbf{p}_2$$에 더 가까워지게 되어 큰 랭킹 손실이 생길 것이다.

**위 예시는 MF가 복잡한 유저-아이템 상호작용을 저차원의 latent space에서 간단하고 고정적인 내적으로 계산하려 할 때 발생할 수 있는 한계를 보여준다.** 한 가지 해결 방법은 $$K$$를 늘리는 것이다. 그러나 이는 sparse한 상황에서는 역으로 모델의 일반화 성능을 저하시킬 수 있다(e.g., 오버피팅). 본 논문에서는 Deep Neural Network를 이용하여 상호작용 함수를 학습시킴으로써 이러한 한계를 다룬다.

## 3. NEURAL COLLABORATIVE FILTERING

### 3.1 General Framework

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

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









90













\






