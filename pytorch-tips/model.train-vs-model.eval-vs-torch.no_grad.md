# model.train() vs model.eval() vs torch.no\_grad()

지금까지 `model.train()`을 한 상태와 `model.eval()`을 한 상태의 파라미터는 똑같을거라 생각하고, 왜 계속 `model.train()`상태에서 pred 뽑을때마다 값이 다르게 나오는지 이해가 안됐음.&#x20;

**`model.train()`에서는 batch normalizaion에서 batch statistics를 이용하고, dropout layer가 활성화된다.**&#x20;

**반면 `model.eval()`을 하면 train과 eval모드에서 각각 다르게 동작해야할 레이어들이 비활성화된다.** batch statics로 결정된 running statistics를 사용하게 되고, dropout layer는 비활성화된다.

`torch.no_grad()`는 그래디언트를 계산하지 않아 메모리 사용량을 줄이고 계산속도를 빠르게 만들 뿐이다.
