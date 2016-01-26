Network Patterns in MLP Gradients
---

No requirements.txt because I think there are prereqs for a lot of this stuff and I was working outside of virtualenv so I would end up dumping a bunch of unrelated stuff into a pip freeze. Numpy, matplotlib, networkx should be all you need.

Gotta have the pickled mnist from theano site (or from [here](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz)), put in the containing folder. To have CIFAR samples, get [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), gunzip and put it in a folder called `cifar-10-batches-py`.
