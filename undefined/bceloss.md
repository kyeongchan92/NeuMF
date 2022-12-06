---
description: Binary Cross Entropy Loss
---

# BCELoss

{% code lineNumbers="true" %}
```python
import torch.nn as nn

# 선언
m = nn.Sigmoid()
criterion = nn.BCELoss()

# 사용
loss = criterion(m(output), label)
loss.backward()
```
{% endcode %}

output은 모델의 아웃풋이다.

$$l(\hat{y}, y) = -w_n [y \log \hat{y} + (1-y)\log (1- \hat{y})]$$

$$w_n$$은 샘플마다 가중치를 줄 수 있다는 것인데, 일단 무시하자.

## $$\hat{y}$$ : 모델의 output <a href="#haty-output" id="haty-output"></a>

$$\hat{y}$$를 모델의 아웃풋이라고 하자. 이 값은 0 \~ 1 사이의 값을 갖는다. 보통의 이진 분류 문제에서는, 마지막에 Sigmoid 계층을 통과시켜 확률처럼 사용한다.

{% code lineNumbers="true" %}
```python
import torch
output = torch.randn(3, requires_grad=True)
output

: tensor([1.8618, 1.6638, 2.3006], requires_grad=True)
```
{% endcode %}

배치사이즈가 3이라고 하자. 모델 output을 위 처럼 1차원으로 뽑을 수도 있고, 2차원으로 뽑을 수도 있다. 보기 쉽게 2차원으로 사용해보자.

```python
output = output.view(-1, 1)
output

: tensor([[1.8618],
          [1.6638],
          [2.3006]], grad_fn=<ViewBackward0>)
```

Sigmoid를 적용시켜보자.

```python
sigmoid_output = m(output)
sigmlid_output

:tensor([[0.8655],
         [0.8407],
         [0.9089]], grad_fn=<SigmoidBackward0>)
```



## $$y$$ : 정답 라벨

$$y$$는 모델이 맞히고자 하는 정답 라벨이다.

```python
label = torch.empty(3).random_(2)
label

: tensor([0., 1., 0.])
```

임의로 생성해봤다. 이 역시 보기 쉽게 2차원으로 바꿔보자.

```python
label = label.view(-1, 1)
label

:tensor([[0.],
         [1.],
         [0.]])
```



### Loss 구하기 <a href="#loss" id="loss"></a>

```python
criterion = nn.BCELoss(reduction='none')
criterion(sigmoid_output, label)

:tensor([[2.0062],
         [0.1735],
         [2.3961]], grad_fn=<BinaryCrossEntropyBackward0>)
```

BCELoss 선언 시 reduction='none'이라는 옵션을 주었는데, 기본값은 'mean'이며 지금은 설명을 위해 'none'으로 줬다.

첫 번째 값인 2.0062를 보자. 이 값은 맨 위 수식에서 l(0.8655,0)의 결과인 것이다. 직접 확인해보자.

```python
import math
-math.log(1 - 0.8655)

: 2.0061910799402436
```



### reduction 기본 옵션은 'mean' <a href="#reduction-mean" id="reduction-mean"></a>

모델 학습 시 샘플별로 loss를 구할 필요는 없다. 보통 평균을 취한 후, 스칼라값을 가지고 역전파시킨다. 그래서 아무 옵션을 주지 않고 구하면

```python
criterion = nn.BCELoss()
criterion(sigmoid_output, label)

: tensor(1.5253, grad_fn=<BinaryCrossEntropyBackward0>)
```

즉, 이 값은 각 샘플별 loss의 평균을 취한 값이다.

```python
import numpy as np
np.mean([2.0062, 0.1735, 2.3961])

: 1.525266666666667
```















