# PyTorch-DANN

A pytorch implementation for paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*

    InProceedings (icml2015-ganin15)
    Ganin, Y. & Lempitsky, V.
    Unsupervised Domain Adaptation by Backpropagation
    Proceedings of the 32nd International Conference on Machine Learning, 2015

## Environment

- Python 2.7
- PyTorch 0.3.1

## Result

results of the default `params.py`

|                                    | MNIST (Source) | USPS (Target) |
| :--------------------------------: | :------------: | :-----------: |
| Source Classifier                  |   99.140000%   |  83.978495%   |
| DANN                               |                |  97.634409%   |

## Credit

- <https://github.com/fungtion/DANN>
- <https://github.com/corenel/torchsharp>