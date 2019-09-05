# PyTorch-DANN

A PyTorch implementation for paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*

    InProceedings (icml2015-ganin15)
    Ganin, Y. & Lempitsky, V.
    Unsupervised Domain Adaptation by Backpropagation
    Proceedings of the 32nd International Conference on Machine Learning, 2015

## Environment

- Python 3.6
- PyTorch 1.0

## Note

- `MNISTmodel()`
    - basically the same network structure as proposed in the paper, expect for adding dropout layer in feature extractor
    - large gap exsits between with and w/o dropout layer
    - better result than paper
- `SVHNmodel()`
    - network structure proposed in the paper may be wrong for both 32x32 and 28x28 inputs
    - change last conv layer's filter to 4x4, get similar(actually higher) result
- `AlexModel`
    - not successful, mainly due to the preprain model difference

## Result

|                      | MNIST-MNISTM   | SVHN-MNIST | Amazon-Webcam |Amazon-Webcam10 |
| :------------------: | :------------: | :--------: | :-----------: |:-------------: |
| Source Only          |   0.5225       |  0.5490    |  0.6420       | 0.             |
| DANN(paper)          |   0.7666       |  0.7385    |  0.7300       | 0.             |
| This Repo Source Only|   -            |  -         |  -            | 0.             |
| This Repo            |   0.8400       |  0.7339    |  0.6528       | 0.             |

## Credit

- <https://github.com/fungtion/DANN>
- <https://github.com/corenel/torchsharp>
- <https://github.com/corenel/pytorch-starter-kit>