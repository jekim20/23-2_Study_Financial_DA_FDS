## 1. 데이터셋 전처리
## 2. IsolationForest과 EllipticEnvelope 이용
#### IsolationForest
- 특히 고차원 데이터셋에서 이상치 감지를 위한 효율적인 알고리즘
- 무작위로 성장한 결정 트리로 구성된 랜덤 포레스트를 만듦 -> 각 노드에서 랜덤하게 선택한 다음 (최솟값과 최댓값 사이에서) 랜덤한 임계값을 골라 데이터셋을 둘로 나눔
===> 이런 식으로 데이터셋은 점차 분리되어 모든 샘플이 다른 샘플과 격리될 때까지 진행됨
- 이상치는 일반적으로 다른 샘플과 멀리 떨어져 있으므로 (모든 결정 트리에 걸쳐) 평균적으로 정상 잼플과 적은 단계에서 격리됨

#### EllipticEnvelope
- Gaussian distributed dataset에서 이상치를 탐지
- scikit-learn은 데이터에 robust 공분산 추정을 fit하는 covariance.EllipticEnvelope 객체를 제공함 => 중앙 데이터 포인트들에 타원을 fit하고 중앙 모드 외부의 포인트들을 무시함
- 예를 들어, 인라이어 데이터가 Gaussian distributed라고 가정하면, 인라이어의 위치와 공분산을 robust하게 추정함(즉, 아웃라이어의 영향을 받지 않고). 이 추정에서 얻은 마할라노비스 거리는 outlyingness의 측정을 얻을 때 사용됨
  
## 3. 목표: GANomaly 이용하여 앙상블 사용
#### GANomaly
- 논문 "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training"
- **encoder-decoder-encoder** architectural model for general anomaly detection enabled by an adversarial training framework
- formulate objective function by combining three loss functions, each of which optimizes individual sub-networks (Adversarial Loss, Contextual Loss, Encoder Loss)
- Ganomaly Pipeline(two encoders, a decoder, and discriminator networks, employed within three sub-networks)
  1. First sub-network is a bow tie autoencoder network behaving as the generator part of the model. The generator learns the input data representation and reconstructs the input image via the use of an encoder and a decoder network, respectively. The formal principle of the sub-network is the following: The generator G first reads an input image X, where X ∈ R a×b , and forward-passes it to its encoder network GE. With the use of convolutional layers followed by batch-norm and leaky ReLU() activation, respectively, GE downscales X by compressing it to a vector z, where z ∈ R d . z is also known as the bottleneck features of G and hypothesized to have the smallest dimension containing the best representation of X. The decoder part GD of the generator network G adopts the architecture of a DCGAN generator [31], using convolutional transpose layers, ReLU() activation and batch-norm together with a tanh layer in the end. This approach upscales the vector z 4 to reconstruct the image X as Xˆ. Based on these, the generator network G generates image Xˆ via Xˆ = GD(z), where z = GE(X).
  2. The second sub-network is the encoder network E that compresses the image Xˆ that is reconstructed by the network G. With different parametrization, it has the same architectural details as GE. E downscales Xˆ to find its feature representation zˆ = E(Xˆ) (Figure 2B). The dimension of the vector zˆ is the same as that of z for consistent comparison.
  3. The third sub-network is the discriminator network D whose objective is to classify the input X and the output Xˆ as real or fake, respectively. This sub-network is the standard discriminator network introduced in DCGAN

## 4. 참고 문헌
- 핸즈온 머신러닝 2판
- https://scikit-learn.org/stable/modules/outlier_detection.html
- 논문 "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training"
