# LightGCN(2020)(작성중)

[paper](https://dl.acm.org/doi/pdf/10.1145/3397271.3401063)

He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020, July). Lightgcn: Simplifying and powering graph convolution network for recommendation. In _Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval_ (pp. 639-648).

[paperswithcode](https://paperswithcode.com/method/lightgcn)

그림은 직접 그림.

## 3 METHOD

이전 섹션에서 NGCF가 Collaborative Filtering에 쓰이기엔 무거운 GCN 모델이라는 것을 증명했다. 이에 기반하여, GCN 중 핵심적인 요소만 이용하여 가볍지만 효과적인 모델을 목표로 삼았다. 모델이 간단해짐으로써 얻는 이점은 여러가지가 있다. 좀 더 해석 가능한 점, 학습과 유지보수가 쉽다는 점, 모델의 수행을 분석하기가 쉽다는 점, 더 효율적인 방법으로 고치기 쉬워진다는 점 등이 있다.

본 섹션에서는 Figure 2에 그려진 것과 같이 Light Convolution Network (LightGCN) 모델 설계를 설명한다.&#x20;

### 3.1 LightGCN

{% hint style="info" %}
용어 정리

<mark style="background-color:red;">**smoothing**</mark> : GCN에서 노드는 이웃 노드를 aggregation하여 업데이트 되는데, 레이어가 깊어질수록 그 범위가 커져서 노드들간의 차이가 점점 사라진다. over-smoothing이란, 레이어가 많아질수록 다수의 노드가 유사해지는 현상을 말한다.
{% endhint %}

GCN의 기본 아이디어는 그래프에서의 특징을 <mark style="background-color:red;">**smoothing**</mark>함으로써 노드의 표현을 학습하는 것이다 \[23, 40]. 이를 위해서, GCN은 반복적으로 graph convolution을 수행한다. 즉, 타겟 노드의 새로운 표현으로서 이웃의 피쳐를 aggregating한다. 이러한 이웃 aggregation은 다음과 같이 요약될 수 있다:

$$
\mathbf{e}_u^{k+1}=\text{AGG}(\mathbf{e}_u^{(k)}, \left\{\mathbf{e}_i^{k}:i\in\mathcal{N}_u \right\}
$$

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption><p>aggregation function</p></figcaption></figure>

AGG라는 것은 **aggregation 함수**(graph convolution의 핵심임!)라는 것을 의미한다. aggregation 함수는 타겟 노드와 그 타겟노드의 이웃 노드들의 $$k$$번째 레이어의 표현을 인풋으로 받는다. 많은 연구가 AGG를 구체화했는데, GIN\[42]에서는 weighted sum을, GraphSAGE\[14]에서는 LSTM aggregator를, BGNN\[48]에서는 bilinear interaction aggregator를 도입했다. 그러나, 대부분의 연구들은 특징 변형 또는 비선형 활성화를 AGG 함수와 결합하였다. 비록 semantic 인풋 피쳐를 갖는 노드 또는 그래프 분류 태스크에서는 그들의 성능이 좋을지라도, collaborative filtering에 대해서는 무거울 수 있다.

#### 3.1.1 Light Graph Convolution (LGC).

LightGCN에서는 간단한 weighted sum aggregator를 사용하고 피쳐 변형과 비선형 활성화 사용을 포기했다. LightGCN에서의 Graph convolution 작업(즉, 순전파\[39])는 다음과 같이 정의된다:

$$
\mathbf{e}_u^{(k+1)}=\sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}}\mathbf{e}_i^{(k)}

\\
\;
\\
\mathbf{e}_i^{(k+1)}=\sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i|}\sqrt{|\mathcal{N}_u|}}\mathbf{e}_u^{(k)}
\\
\;
\\
\text{equation (3)}
$$

대칭적인 정규화항(symmetric normalization term) $$\frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}}$$는 graph convolution 작업 시 임베딩의 스케일이 커지는 것을 막는 기본 GCN의 디자인 \[23]을 따른다. $$L_1$$같은 다른 옵션이 사용될 수도 있지만, 경험적으로 이 대칭적 정규항이 좋은 성능을 보였다 (4.4.2의 실험 결과를 참고).

LGC에서 알고 있으면 좋은 점은 바로, 오직 연결된 이웃들만 aggregate하고, 자기 자신은 통합(셀프커넥션)하지 않는다는 것이다. 이 부분은 기존의 대부분의 graph convolution 작업 \[14, 23, 36, 39, 48]이 더 멀리 있는 이웃들을 aggregate하고 셀프커넥션을 특별하게 다루었던 점과 다르다. 다음 서브섹션에서 말할 것이지만, 레이어 결합 과정은 기본적으로 셀프커넥션과 같은 효과를 낸다. 그러므로, LGC에서 셀프커넥션을 포함할 필요는 없다.

#### 3.1.2 Layer Combination and Model Prediction.

LightGCN에서 학습되는 모델 파라미터들은 오직 0번째 레이어에서 의 임베딩이다. 즉, 유저에 관한 $$\mathbf{e}_u^{(0)}$$, 아이템에 관한 $$\mathbf{e}_i^{(0)}$$이다. 이들이 주어지면, 높은 레이어의 임베딩들은 equation (3)에서 정의된 LGC를 통해 계산된다. K개의 레이어를 통과한 후엔 각 레이어에서 얻어진 임베딩을 결합하여 유저 또는 아이템의 최종 표현을 생성한다.

$$
\mathbf{e}_u = \sum_{k=0}^{K}\alpha_K \mathbf{e}_u^{(k)};\;\;\;\mathbf{e}_i = \sum_{k=0}^{K}\alpha_K \mathbf{e}_i^{(k)}

\\
\;
\\
\text{equation (4)}
$$

$$\alpha_k \ge 0$$은 최종 임베딩을 구성에 참여하는 $$k$$번째 레이어 임베딩의 중요도를 의미한다. $$\alpha_k$$는 하이퍼파라미터로서 직접 설정할 수도 있고, 모델 파라미터(e.g. 어텐션 네트워크\[3]의 아웃풋)로 설정해서 자동으로 최적화되도록 할 수도 있다. 본 논문의 실험에서는 $$\alpha_k$$를 $$1/(K+1)$$로 uniformly하게 설정하는 것이 일반적으로 좋은 성능을 보였다. 그러므로 우리는 LightGCN이 불필요하게 복잡해지는 것을 피하기 위하여 $$\alpha_k$$를 최적화 하기 위한 특별한 요소를 설계하지 않았다. 최종 임베딩을 만들 때 레이어들을 결합하는 이유는 3가지이다. (1) 레이어 수가 증가할수록 임베딩은 **over-smoothed**된다\[27]. 그러므로 마지막 레이어의 임베딩만 사용하는 것은 문제가 있다. (2) 다른 레이어의 다른 임베딩은 다른 semantic을 갖는다. 예를 들어, 첫 번째 레이어는 상호 작용이 있는 유저와 아이템에 smoothness를 적용하고, 두 번째 레이어는 상호 작용한 아이템(유저)을 공유하는 유저(사용자)를 smooth하며, 높은 레이어는 고차원의 연결을 캐치한다\[39]. 그러므로 이들을 결합한 표현은 더욱 종합적인 이해능력을 갖는다. (3) Weighted sum 방식으로 여러 레이어의 임베딩을 결합하면 GCN에서 중요한, 셀프커넥션을 갖는 graph convolution의 효과를 가질 수 있다(3.2.1에서 증명).

모델 얘측은 유저 최종 임베딩과 아이템 최종 임베딩의 내적으로 정의된다.

$$
\hat{y}_{ui}=\mathbf{e}_u^T\mathbf{e}_i

\\
\;
\\
\text{equation (5)}
$$

추천 목록을 생성할 때 이 숫자가 랭킹 스코어로 사용된다.&#x20;

#### 3.1.3 Matrix Form

계산 과정을 촉진시켜줄 LightGCN의 matrix form에 대해 소개한다. 유저-아이템 상호작용 행렬을 $$\mathbf{R}\in \mathbb{R}^{M \times N}$$이라고 하고 $$M$$과 $$N$$이 각각 유저의 수, 아이템의 수를 나타낸다고 하자. 그리고 각각의 엔트리 $$R_{ui}$$는 $$u$$와 $$i$$간의 상호작용이 있으면 1이고 없으면 0이다. 이에 따라 유저-아이템 그래프의 인접행렬을 다음과 같이 얻는다:

$$
\mathbf{A}
=\begin{pmatrix} \mathbf{0}
 & \mathbf{R}
 \\ \mathbf{R}^T
 & \mathbf{0}
 \end{pmatrix}

\\

\;

\\

\text{equation (6)}
$$

$$T$$는 임베딩 사이즈라고 할 때, **** 0번째 레이어를 임베딩 행렬 $$\mathbf{E}^{(0)} \in \mathbb{R}^{(M+N)\times T}$$라고 하자. 이에 따라 LGC와 동일한 형태의 행렬을 얻을 수 있다:

$$
\mathbf{E}^{(k+1)}=(\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}})\mathbf{E}^{(k)}

\\
\;
\\
\text{equation (7)}
$$

$$\mathbf{D}$$는 $$(M+N) \times (M+N)$$의 대각행렬이며, 각 엔트리 $$D_{ii}$$는 $$\mathbf{A}$$의 $$i$$번째 row의 0이 아닌 엔트리의 수이다. Degree 행렬이라고도 부른다. 마지막으로, 예측을 위한 최종 임베딩 행렬을 다음과 같이 얻을 수 있다:

$$
\begin{align*}

\mathbf{E}&=\alpha_0\mathbf{E}^{(0)} +\alpha_1\mathbf{E}^{(1)} + \cdots + \alpha_K\mathbf{E}^{(K)}
\\
&=\alpha_0\mathbf{E}^{(0)} + \alpha_1\tilde{\mathbf{A}}\mathbf{E}^{(0)} + \alpha_2\tilde{\mathbf{A}}^2\mathbf{E}^{(0)}
+\cdots+
\alpha_K\tilde{\mathbf{A}}^K\mathbf{E}^{(0)}
\end{align*}
\\ \; \\
\text{equation (8)}
$$

$$\tilde{\mathbf{A}}=\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}$$는 대칭의 정규화된 행렬이다.

### 3.2 Model Analysis

LightGCN의 간단한 구조 뒤의 합리성을 증명하기 위해 모델 분석을 수행한다. 춧 번째로 Simplified GCN (SGCN)\[40]과의 연결에 대해 논의할 것인데, 이 모델은 최신의 GCN 모델이며 셀프커넥션을 graph convolution에 통합한 모델이다; 본 분석에서는 레이어를 결합함으로써 LightGCN이 셀프 커넥션의 효과를 가질 수 있다는 것, 그러므로 LightGCN은 인접행렬 속에서 셀프커넥션을 추가할 필요가 없다는 점을 보인다. 그 후 Personalized PageRank\[15]로부터 영감을 받아 oversmoothing을 다룬 최신 GCN 변형인 Approximate Personalized Propagation of Neural Predictions (APPNP)\[24]와의 관련성에 대해 논의한다; 본 분석은 LightGCN과 APPNP 사이의 기본적인 동일성을 보이고, 그러므로 LightGCN이 컨트롤이 가능한 oversmoothing과 함께 긴 범위를 순전파 함의 이점을 똑같이 누린다는 것을 보인다. 마지막으로 두 번째 레이어를 분석하여 LGC가 어떻게 한 유저의 2차원 이웃노드까지 smooth할 수 있는지 보인다.

#### 3.2.1 Relation with SGCN

생략

#### 3.2.2 Relation with APPNP

생략

#### 3.2.3 Second-Order Embeding Smoothness

생략

## 3.3 Model Training

LightGCN의 학습가능한 파라미터는 오직 0번째 레이어의 임베딩, 즉 $$\Theta=\left\{ \mathbf{E}^{(0)} \right\}$$뿐이다. 즉, 모델의 복잡도는 표준 matrix factorization(MF)할 때와 똑같다. 우리는 미관측된 엔트리보다 관측된 엔트리의 예측값을 더욱 권장하는 pairwise loss인 BPR(Bayesian Personalized Ranking) loss\[32]를 사용했다.

$$
L_{BPR}=-\sum_{u=1}^M \sum_{i \in \mathcal{N}_u} \sum_{j \notin \mathcal{N}_u} \ln \sigma(\hat{y}_{ui}-\hat{y}_{uj}) + \lambda ||\mathbf{E}^{(0)}||^2
$$

$$\lambda$$는 $$L_2$$ 정규화 텀의 강도 조절 요인이다. 미니배치 형식으로 Adam\[22] 옵티마이저를 사용했다. Hard negative sampling\[31]이나 adversarial sampling\[9]같은, LightGCN의 학습을 향상시킬 수 있는 진보된 네거티브 샘플링을 알고 있지만, 이는 본 연구의 초점이 아니므로 추후 연구로 남겨놓겠다.

GCN과 NGCF에서 사용되는 드랍아웃은 사용하지 않았다. 그 이유는 LightGCN은 피쳐를 transformation하는 가중치 행렬이 없어서 임베딩 레이어에 $$L_2$$ 정규화로 강제하는 것으로도 오버피팅을 충분히 막을 수 있기 때문이다. 이는 LightGCN의 간단함으로부터 나온 것이다 - 두 개의 드랍아웃 ratio(노드 드랍아웃, 메시지 드랍아웃)을 조절해야하고 각 레이어의 임베딩을 정규화(normalize)까지 추가적으로 해야하는 NGCF보다 학습시키기 쉽다.

게다가,&#x20;









\[9] Jingtao Ding, Yuhan Quan, Xiangnan He, Yong Li, and Depeng Jin. 2019. Reinforced Negative Sampling for Recommendation with Exposure Data. In IJCAI. 2230–2236

\[31] Steffen Rendle and Christoph Freudenthaler. 2014. Improving pairwise learning for item recommendation from implicit feedback. In WSDM. 273–282.

\[32] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In UAI. 452– 461.







