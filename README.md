# PyTorch-DANN

A pytorch implementation for paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*

    InProceedings (icml2015-ganin15)
    Ganin, Y. & Lempitsky, V.
    Unsupervised Domain Adaptation by Backpropagation
    Proceedings of the 32nd International Conference on Machine Learning, 2015

## Environment

- Python 2.7/3.6
- PyTorch 0.3.1post2

## Result

results of the default `params.py`

|                                    | SVHN (Source)  | MNIST (Target)|
| :--------------------------------: | :------------: | :-----------: |
| Source Classifier                  |   92.92%   |  68.66%   |
| DANN                               |                |  ----%   |

## Other implementations

- authors(caffe) <https://github.com/ddtm/caffe>
- TensorFlow, <https://github.com/pumpikano/tf-dann>
- Theano, <https://github.com/shucunt/domain_adaptation>
- PyTorch, <https://github.com/fungtion/DANN>

## Credit

- <https://github.com/fungtion/DANN>
- <https://github.com/corenel/torchsharp>
- <https://github.com/corenel/pytorch-starter-kit>