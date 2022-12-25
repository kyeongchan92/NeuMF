# Implicit data에서의 nDCG

출처 : [He, X., Chen, T., Kan, M. Y., & Chen, X. (2015, October). Trirank: Review-aware explainable recommendation by modeling aspects. In _Proceedings of the 24th ACM international on conference on information and knowledge management_ (pp. 1661-1670).](https://dl.acm.org/doi/pdf/10.1145/2806416.2806504)

## 4. EXPERIMENTS > Evaluation Metrics.

$$
NDCG@K=Z_K \sum_{i=1}^{K} \frac{2^{r_i}-1}{\log_2(i+1)}
$$

$$Z_K$$는 제일 좋은 성능일 때 1로 만들기 위한 normalizer이다. $$r_i$$는 $$i$$번째 아이템의 graded relavance이다. (implicit data이기 때문에) $$r_i$$는 1 또는 0이며, 1일 때는 아이템이 test set에 존재할 때이고, 0일 때는 그렇지 않을 때이다.

예를 들어 $$K=25$$라고 해보자.

