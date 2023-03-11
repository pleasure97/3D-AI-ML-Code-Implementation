## NeRF : Representing Scene as Neural Radiance Fields for View Synthesis 


### 논문의 연구 목적 
---


 논문의 제목에서 알 수 있듯이 논문은 View Synthesis를 목표하고 있다. 
 </br>
 View Synthesis는  complex scenes과 같은 입력 데이터를 활용해, 새로운 관점에서 바라본 scene을 생성하는 작업이다. 
 </br>
 </br>
![Figure 1 : Synthetic Drums Scene](./img/NeRF-1.png)
</br>
</br>
NeRF는 드럼을 100가지 관점에서 바라본 입력 데이터들을 활용해, 기존의 관점과 다른 두 가지의 관점에서 바라본 드럼에 대한 scene을 생성한다.
</br>
</br>
### 선행 연구의 한계
---
NeRF라는 모델이 나오기 전에 View Synthesis를 다루는 모델은 크게 두 부류였다.
</br>
</br>
&nbsp; 첫째, 연속적인 3D shape을 사용하는 implicit representation인 'Neural 3D shape representations'이다. implicit representation이 무엇인지 먼저 알아보면, implicit representation은 연속적인 위치 좌표가 입력값이고, 위치 좌표의 scene property가 출력값인 함수의 형식이다. 반면, image, mesh, point cloud와 같은 explicit representation은 픽셀과 같은 불연속적인 위치 좌표를 입력값으로 받는다. explicit representation과 비교해볼 때,  implicit representation은 연속적인 위치 좌표를 계산함으로써 표현할 수 있는 signal이 연속적이게 된다. 따라서, implicit representationn은 neural network의 뛰어난 표현력에 적합한 형식이다. 
</br>
&nbsp; implicit representation인 Neural 3D shape representations은 위와 같은 장점이 있지만, xyz라는 3D 위치 좌표만을 사용하기 때문에 단점도 있다. xyz라는 3D 위치 좌표만을 입력값으로 받게 되면 단순한 shape와 낮은 geometric complexity에 제한되어 렌더링이 oversmooth되는 경향이 있다. NeRF는 선행연구의 한계를 극복하기 위해 5D radiance fields를 사용해 더 높은 해상도를 지닌 렌더링을 생성한다.
 </br>
 </br>
 &nbsp; 둘째, RGB image들을 활용해 높은 quality의 view synthesis를 목표로 하는 'sampled volumetric representations'이다. sampled volumetric representations은 다양한 형태와 재질을 표현할 뿐만 아니라, gradient 기반의 optimization에도 잘 맞는다. 최근의 sampled volumetric representation은 각 image로부터 voxel grids를 sample하고, 낮은 해상의 voxel grids의 불연속성으로 발생하는 noise들을 처리할 수 있는  CNN 구조를 지닌다. sampled volumetric representation은 위와 같은 이산적인 sampling으로 인해 고해상도의 이미지에서는 더 정교한 sampling이 요구되어, 처리 시간이 길어지는 단점도 있다. NeRF는 연속적인 volume을 fully-connected neural network에 인코딩하여 더 높은 해상의 렌더링을 더 낮은 비용으로 생성해낼 수 있다.
 </br>
 </br>
### NeRF Architecture 
---
 ![Fig 2 : An Overview of NeRF Architecture](./img/NeRF-2.png)
 </br>
 </br>
 &nbsp; NeRF는 View Synthesis와 관련된 선행 연구들의 한계점을 극복하기 위한 세 가지 핵심 모듈이 있다.
 </br>
 &nbsp; 첫째, 5D Neural Radiance Fields & MLP network이다. 3D 위치 벡터와 2D viewing direction 벡터를 모델의 입력으로 사용하고, 모델의 입력을 MLP에서 처리해 연속적인 장면들의 복잡한 기하학적 특징(e.g., density)을 추출한다.
 </br>
 &nbsp; 둘째, Classical Volume Rendering & Stratified Sampling이다. 기존의 Volume Rendering 방식을 사용해, 추출된 volume의 density를 camera ray의 color로 렌더링한다. 그리고 렌더링된 camera ray의 color vector, $C(\vec{r})$과 비교할 NeRF에서 예측할 color vector, $\hat{C}(\vec{r})$을 만들기 위해 stratified sampling을 사용한다.
</br>
&nbsp; 셋째, Positional Encoding & Hierarchical Volume Sampling이다. Positional Encoding은 MLP가 5D Radiance Field라는 입력의 high-frequency representation을 학습할 수 있도록 돕는다. 그리고 Hierarchical Volume Sampling은 coarse network와 fine network라는 hierarchical representation을 활용해 최종 렌더링에 도움이 될 법한 sample들을 추출해 렌더링을 효율적으로 만든다. 
</br>
&nbsp; 정리해보면, NeRF는 3D location vector와 2D viewing direction vector라는 입력을 받아 View Synthesis에 필요한  color vector와 density를 출력하는 모델이다. 그리고 NeRF는 모델 내부에서 positional encoding, MLP, rendering, sampling 등 다양한 techniques을 활용해 high quality를 갖는 novel view synthesis를 목표로 한다.
</br>
</br>
### Core Components of NeRF
---
&nbsp; 위에서 설명했던 NeRF의 구성 요소들을 구체적으로 살펴보고자 한다.
</br>
&nbsp; 처음으로 설명했던 5D Neural Radiance Fields와 MLP network를 알아본다.
</br>
&nbsp; 먼저, 5D Neural Radiance Fields에 속하는 3D location vector와 2D viewing direction vector는 다음과 같이 표현한다. $$\vec{x} = (x, y, z) ,  \vec{d}= (\theta, \phi)$$
&nbsp; 다음으로, 5D Neural Radiance Fields를 입력으로 받아 color와 density를 출력하는 MLP는 아래와 같이 표현할 수 있다. $$F_\Theta : (\vec{x}, \vec{d}) \to (\vec{c}, \sigma)$$
&nbsp; 그리고 MLP의 구체적인 구조는 다음과 같다.
</br>
</br> 
![Fig 3 : MLP architecture](./img/NeRF-9.png)
</br>
</br>
&nbsp; 두 번째로 설명했던 Classical Volume Rendering과 Stratified Sampling을 살펴본다.
</br>
&nbsp; Classical Volume Rendering과 관련된 수식은 다음과 같다. 
</br>
</br>
![](./img/NeRF-3.png)
</br>
</br>
&nbsp; 하나씩 이해하면서 수식을 더 깊이 이해해보자. 
&nbsp; $t_n :$ camera ray의 가장 가까운 경계 지점 
&nbsp; $t_f:$ camer ray의 가장 먼 경계 지점 
&nbsp; $T(t):$ 축적된 투과율 (= camera ray가 $t_n$에서 $t$까지 지나면서 어떤 입자와도 충돌하지 않을 확률) 
&nbsp; $\sigma(\vec{x}):$ volume density (= 위치 $\vec{x}$에 존재하는 입자로 인해 camera ray가 중단될 확률)
&nbsp; $\vec{r}(t) = \vec{o} + t\vec{d} :$ $\vec{o}$는 ray의 원점, $\vec{d}$는 ray의 방향 벡터
</br>
&nbsp; 종합해보면, $t_n$에서 $t_f$ 사이의 camera ray $\vec{r}$의 color는 camera ray $\vec{r}$의 방향, $\vec{r}$이 입자와 충돌할 확률, $\vec{r}$이 입자와 충돌했을 때 중단될 확률을 고려한 것이다. 
</br>
</br>
![](./img/NeRF-4.png)
</br>
</br>
&nbsp; Stratified Sampling은 MLP가 특정 구간에 쏠린 sample들을 학습하지 않도록 $t_n$에서 $t_f$ 사이를 $N$개의 구간으로 나누어 각 구간에서 한 sample씩 균등하게 추출될 수 있도록 한다. 
</br>
&nbsp;Stratified Sampling으로 추출된 sample들을 활용해 $C(\vec{r})$과 비교할 $\hat{C}(\vec{r})$을 계산한다.
</br>
</br>
![](./img/NeRF-5.png)
</br>
</br>
