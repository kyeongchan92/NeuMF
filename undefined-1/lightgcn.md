# LightGCN(작성중)

[paper](https://dl.acm.org/doi/pdf/10.1145/3397271.3401063)

[paperswithcode](https://paperswithcode.com/method/lightgcn)

## 3 METHOD

이전 섹션에서 NGCF가 Collaborative Filtering에 쓰이기엔 무거운 GCN 모델이라는 것을 증명했다. 이에 기반하여, GCN 중 핵심적인 요소만 이용하여 가볍지만 효과적인 모델을 목표로 삼았다. 간단한 모델의 이점은 여러가지가 있다. 좀 더 해석 가능한 점, 학습과 유지보수가 쉽다는 점, 모델의 수행을 분석하기가 쉽다는 점, 더 효율적인 방법으로 고치기 쉬워진다는 점 등이 있다.

본 섹션에서는 Figure 2에 그려진 것과 같이 Light Convolution Network (LightGCN) 모델 설계를 설명한다.&#x20;

### 3.1 LightGCN

GCN의 기본 아이디어는 그래프에서의 특징을 smoothing함으로써 노드의 표현을 학습하는 것이다 \[23, 40]. 이를 위해서, GCN은 반복적으로 graph convolution을 수행한다. 즉, 타겟 노드의 새로운 표현으로서 이웃의 피쳐를 aggregating한다. 이러한 이웃 aggregation은 다음과 같이 요약될 수 있다:

$$
\mathbf{e}_u^{k+1}=\text{AGG}(\mathbf{e}_u^{(k)}, \left\{\mathbf{e}_i^{k}:i\in\mathcal{N}_u \right\}
$$

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

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

#### 3.2.1 Layer Combination and Model Prediction.

LightGCN에서 학습되는 모델 파라미터들은 오직 0번째 레이어에서 의 임베딩이다. 즉, 유저에 관한 $$\mathbf{e}_u^{(0)}$$, 아이템에 관한 $$\mathbf{e}_i^{(0)}$$이다. 이들이 주어지면, 높은 레이어의 임베딩들은 equation (3)에서 정의된 LGC를 통해 계산된다. K개의 레이어를 통과한 후엔 각 레이어에서 얻어진 임베딩을 결합하여 유저 또는 아이템의 최종 표현을 생성한다.

$$
\mathbf{e}_u = \sum_{k=0}^{K}\alpha_K \mathbf{e}_u^{(k)};\;\;\;\mathbf{e}_i = \sum_{k=0}^{K}\alpha_K \mathbf{e}_i^{(k)}

\\
\;
\\
\text{equation (4)}
$$

$$\alpha_K \ge 0$$은 최종 임베딩을 구성하는 $$k$$번째 레이어의 중요도를 의미한다.

&#x20;



