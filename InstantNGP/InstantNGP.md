## InstantNGP - Instant Neural Graphics Primitives with a Multiresolution Hash Encoding 


### 논문의 연구 목적 
---
</br>

![](./img/InstantNGP-1.png)
</br>
</br>
&nbsp; **InstantNGP는 Neural Graphics Primitives 분야에서 Multiresolution Hash Encoding이라는 Input Encoding을 도입해 최적화된 accuracy와 training, inference speed를 내는 것을 목표한다.**
</br>
</br>
&nbsp; Neural Graphics Primitives에 대해 설명하기에 앞서, Graphics Primitives는 Computer Graphics에서 사용되는 데이터 단위, 혹은 representation을 의미한다. Graphics Primitives는 외적인 모습을 parameterize하는 수학적 함수로 표현된다. Graphics Primitives의 예시로는, 2D Image, 3D Mesh, Multi-Dimensional Radiance Field 등이 있다. Neural Graphics Primitives는  Neural Network를 기반으로 만든 Graphics Primitives를 의미하며, Neural Network로는 주로 Fully-Connected Neural Network가 사용된다. InstantNGP는 적은 수의 Fully-Connected Neural Network를 사용해도 Multiresolution Hash Encoding 덕분에 좋은 성능을 보였다.
</br>
</br>
&nbsp; **InstantNGP는 4가지의 구체적인 Neural Graphics Primitives 분야에서 성과를 보였다.** 
</br>
&nbsp; 첫 번째는, GigaPixel Image Approximation으로 High Resolution Image를 압축하는 분야이다. 
</br>
&nbsp; 두 번째는, Signed Distance Functions (SDF)으로 어떤 점과 객체 간의 거리를 계산해 3D 객체를 생성하고 변형하는 분야이다.
</br>
&nbsp; 세 번째는,  Neural Radiance Caching으로 feature buffers를 활용해 사진과 같이 현실적인 pixel color들을 예측해내는 분야이다.
</br>
&nbsp; 네 번째는, Neural Radiance and Density Fields (NeRF)으로 5D spatial-directional function으로 새로운 관점에서의 이미지를 생성해내는 분야이다.

### 선행 연구의 한계
---
</br>
&nbsp; InstantNGP가 MultiResolution Hash Encoding이라는 input encoding을 도입하기 전에 선행 연구들은 한계점이 명확히 보이는 input encoding들을 사용했다.
</br>

&nbsp; **선행 연구들에서 사용된 input encoding의 종류와 그에 대한 한계를 구체적으로 살펴보고자 한다.**
</br>
</br>
&nbsp; 첫째, linearly separable한 특성을 지닌 encoding이다. 복잡한 형태의 데이터를 선형적으로 분리시키는 초기 버전의 encoding으로, one-hot encoding과 kernel-trick이 대표적이다. 위와 같은 encoding은 자체적으로 sparse해지거나 curse of dimensionality가 발생할 수 있어, 단순히 고차원으로 mapping하는 encoding이라는 한계가 있다.
</br>
</br>
&nbsp; 둘째, frequency encoding이다. 대표적으로는 Transformer의 positional encoding이 있으며, sine 함수와 cosine 함수 같은 주기함수를 사용해 고차원 공간으로 mapping한다. frequency encoding은 mapping할 차원의 수만을 입력값으로 받기 때문에 아래에서 살펴볼 자체 학습 가능한 encoding들보다 accuracy가 떨어지는 단점이 있다.
</br>
</br>
&nbsp; 셋째, parametric encoding이다. parametric encoding은 grid나 tree와 같은 데이터 구조에 학습 가능한 parameter들을 배치하고, input vector에 맞게 parameter들을 interpolate하는 encoding이다.  parametric encoding은 frequency encoding보다 accuracy가 뛰어난 장점이 있지만, 학습 가능한 parameter들을 저장하고 처리하는 데 있어 memory footprint와 computational cost가 크다는 단점도 있다.
</br>
</br>
&nbsp; 넷째, sparse parametric encoding이다. sparse parametric encoding은 parametric encoding의 memory footprint와 computational cost 문제를 해결하기 위해 제안된 encoding이다. sparse parametric encoding은 octree나 sparse grid와 같은 데이터 구조를 사용해 dense grid에서 사용되지 않는 feature들을 제거하거나, coarse stage와 fine stage 같은 multi stage 구조로 feature grid에서 필요한 feature들만 정제하는 encoding이다. 하지만 sparse parametric encoding은 데이터 구조를 변형시키기 때문에 특정 작업에만 specific한 구조를 가지거나, feature grid를 주기적으로 update해야 해서 학습 과정이 복잡해진다.

### Instant NGP
---
</br>

&nbsp; **InstantNGP는 선행 연구들에서 제시된 encoding들의 단점을 보완하기 위해 다음과 같은 특성을 지닌 encoding을 제시했다.**
</br>
</br>

&nbsp; 첫째, 학습 가능한 feature vector들을 spatial hash table에 저장한다. spatial hash table의 크기를 $T$라는 hyperparameter로 받아, parameter의 수와 reconstruction quality를 고려해 spatial hash table의 크기를 조정할 수 있게끔 만들었다.
</br>
</br>
&nbsp; 둘째, resolution의 크기에 따라 spatial hash table를 두고, spatial hash table들의 outputs를 concatenate하여 MLP(Multi Layer Perceptron)에 통과시킨다.
</br>
</br>
&nbsp; 셋째, hash function의 hash collision을 일부러 해결하지 않는다. hash function에 의해 만들어지는 hash table에서는 되도록이면 서로 다른  key가 동일한 hash를 가리키지 않아야 한다. 하지만 hash function은 hash의 수가 제한되어 있지만 무한히 key가 생성될 수 있는 함수이기 때문에, 비둘기집의 원리에 의해 필연적으로 서로 다른 key가 같은 hash를 가리키는 문제가 발생할 수밖에 없다. 이를 hash collision이라고 한다. 그래서 일반적으로 hash function을 사용할 때는 어떻게든 hash collision을 최소화하려고 한다. 하지만 논문의 저자들은 일부러 hash collision을 해결하지 않는다. hash collision을 해결하지 않아도 spatial hash table들의 output vector들이 MLP에 통과되면, MLP가 backpropgation 과정에서 update에 중요한 hash entry 구분해내기 때문이다.
</br>
</br>
&nbsp; **Multiresolution Hash Encoding의 특징에 따라 다음과 같은 장점을 지닌다.**
</br>
</br>
&nbsp; 첫째, training speed와 inference speed가 빨라진다. hash table entry에 대한 접근 속도가 빠를 뿐만 아니라, hash table 자체가 parallelization 연산에 특화되어 있기 때문이다.
</br>
</br>
&nbsp; 둘째, 특정한 task에 specific해질 수 있는 문제를 예방한다. Multiresolution hash encoding은 데이터 구조에 대한 특수한 변형이 없기 때문에 다양한 task들에 대해 일반적인 적용이 가능할 뿐만 아니라, 불필요한 control flow를 없애 GPU를 더 효율적으로 쓸 수 있도록 한다.


### Core Components of InstantNGP
---
</br>

![](./img/InstantNGP-2.png)
</br>
</br>

### Technical details of InstantNGP
---
### 논문의 한계 및 배울 점 
---
