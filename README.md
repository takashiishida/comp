# Complementary-Label Learning

This repository gives the implementation for *complementary-label learning* from the ICML 2019 paper [1], the ECCV 2018 paper [2], and the NeurIPS 2017 paper [3].

## Requirements
- Python 3.6
- numpy 1.14
- PyTorch 1.1
- torchvision 0.2

## Demo
The following demo will show the results with the MNIST dataset.  After running the code, you should see a text file with the results saved in the same directory.  The results will have three columns: epoch number, training accuracy, and test accuracy.

```bash
python demo.py -h
```

#### Methods and models
In `demo.py`, specify the `method` argument to choose one of the 5 methods available:

- `ga`: Gradient ascent version (Algorithm 1) in [1].
- `nn`: Non-negative risk estimator with the max operator in [1].
- `free`: Assumption-free risk estimator based on Theorem 1 in [1].
- `forward`: Forward correction method in [2].
- `pc`: Pairwise comparison with sigmoid loss in [3].

Specify the `model` argument:

- `linear`: Linear model
- `mlp`: Multi-layer perceptron with one hidden layer (500 units)

## Reference
1. T. Ishida, G. Niu, A. K. Menon, and M. Sugiyama.<br>**Complementary-label learning for arbitrary losses and models**.<br>In *ICML 2019*.<br>[[paper]](https://arxiv.org/abs/1810.04327)
2. Yu, X., Liu, T., Gong, M., and Tao, D.<br>**Learning with biased complementary labels**.<br>In *ECCV 2018*.<br>[[paper]](https://arxiv.org/abs/1711.09535)
3. T. Ishida, G. Niu, W. Hu, and M. Sugiyama.<br>**Learning from complementary labels**.<br>In *NeurIPS 2017*.<br>[[paper]](https://arxiv.org/abs/1705.07541)

If you have any further questions, please feel free to send an e-mail to: ishida at ms.k.u-tokyo.ac.jp.
