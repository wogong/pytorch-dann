# PyTorch-DANN

A pytorch implementation for paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*

    InProceedings (icml2015-ganin15)
    Ganin, Y. & Lempitsky, V.
    Unsupervised Domain Adaptation by Backpropagation
    Proceedings of the 32nd International Conference on Machine Learning, 2015

## Environment

- Python 3.6
- PyTorch 0.4.1

## Note

- `Config()` 为针对特定任务的配置参数。
- `MNISTmodel()` 完全按照论文中的结构，但是 feature 部分添加了 `Dropout2d()`，实验发现是否添加 `Dropout2d()` 对于最后的性能影响很大。最后实验重现结果高于论文，因为使用了额外的技巧，这里还有值得探究的地方。
- `SVHNmodel()` 无法理解论文中提出的结构，为自定义结构。最后实验重现结果完美。
- MNIST-MNISTM: `python mnist_mnistm.py`
- SVHN-MNIST: `python svhn_mnist.py`
- Amazon-Webcam: `python office.py` 没有复现成功

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