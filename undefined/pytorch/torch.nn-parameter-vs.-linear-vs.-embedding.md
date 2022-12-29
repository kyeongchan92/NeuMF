# torch.nn: Parameter vs. Linear vs. Embedding

[Audrey Wang](https://audreywongkg.medium.com/?source=post\_page-----2131e319e463--------------------------------)[Insight into PyTorch.nn: Parameter vs. Linear vs. Embedding](https://audreywongkg.medium.com/pytorch-nn-parameter-vs-nn-linear-2131e319e463)를 번역하였습니다.

**torch.nn.Parameter()** 클래스를 사용하면 초기값이 _1.4013e-45_처럼 매우 작다. 이로 인해 결과가 매우 이상해지기도 한다. **torch.nn.Parameter()** 대신 **torch.nn.Linear()**를 사용하면 괜찮은 값으로 초기화되는 것을 볼 수 있다.

정확히 **nn.Parameter()**와 **nn.Linear()**가 수행하는 일은 무엇일까? 또한, nn.Embedding()과의 차이는 무엇일까?

## nn.Parameter

```python
weight = torch.nn.Parameter(torch.FloatTensor(2,2))
```

이 코드는 **nn.Parameter()** 사용하여 파라미터를 생성한 예시이다. `weight`는 주어진 텐서에 의해 생성되었는데, 즉, weight의 초기값은 torch.FloatTensor(2,2)가 된 것이다. 매우 작은 값이었던 이유는 torch.FloatTensor(2,2)가 매우 작았기 때문이다. torch.FloatTensor(2,2)는 어떤 값일까?

```python
a = torch.FloatTensor(2,2)
print(a)

>> tensor([[4.6837e-39, 9.9184e-39],
           [9.0000e-39, 1.0561e-38]])
```

0에 가까운 매우 작은 값이다.





****

## nn.Linear <a href="#680c" id="680c"></a>



## nn.Embedding

