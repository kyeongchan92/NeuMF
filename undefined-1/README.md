# 딥러닝을 이용한 추천시스템

[Naumov, M., Mudigere, D., Shi, H. J. M., Huang, J., Sundaraman, N., Park, J., ... & Smelyanskiy, M. (2019). Deep learning recommendation model for personalization and recommendation systems. arXiv preprint arXiv:1906.00091.(페이스북, 인용수 327)](https://arxiv.org/pdf/1906.00091.pdf)



CTR 예측 및 랭킹시스템을 포함한 <mark style="background-color:blue;">**개인화(Personalization)**</mark> 및 <mark style="background-color:blue;">**추천시스템(recommandation system)**</mark>은 현재 다양한 분야에서 많은 기업들에서 사용되고 있다. **이들은 오래된 역사를 갖고 있음에도 불구하고, neural network를 포함하게 된 지는 얼마 되지 않았다.** 개인화 및 추천시스템을 위한 딥러닝 모델의 아키텍쳐 설계에는 다음 두 가지의 주요 관점이 기여했다.

* 첫 번째는 추천시스템 관점이다. 초기에는 전문가들이 카테고리에 따라 아이템을 분류하는 방식을 사용했다. 이 방식은 이후 Collaborative Filtering(CF)로 발전하여 평점 등의 유저 행동을 사용하게 되었다. 이 방식은 유저 또는 아이템을 그룹핑하는 Neighborhood 방법, 행렬분해를 사용하는 Latent factor 방법으로 세분화되었다.
* 두 번째는 주어진 데이터에 기반하여 이벤트를 분류하거나 확률을 예측하기 위해서 통계모델 이용하는, 즉 예측의 관점이다 \[5]. 예측 모델은 선형회귀, 로지스틱 회귀\[26] 등의 간단한 모델에서 딥 네트워크를 통합한 모델로 변모해갔다. 이 모델들은 카테고리컬 데이터를 처리하기 위하여 원핫, 멀티핫 벡터를 추상 공간속의 밀집표현으로 변화시키는, 즉 임베딩을 사용하는 방법을 채택했다\[20]. 추상 공간이란 추천시스템에 의해 발견된, **latent factor의 공간**으로 해석될 수 있다.
