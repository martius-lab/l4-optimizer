# L4 Stepsize Adaptation Scheme

By [Michal Rolínek](https://scholar.google.de/citations?user=DVdSTFQAAAAJ&hl=en), [Georg Martius](http://georg.playfulmachines.com/).

[Autonomous Learning Group](https://al.is.tuebingen.mpg.de/), [Max Planck Institute for Intelligent Systems](https://is.tuebingen.mpg.de/).

## Table of Contents
0. [Introduction](#introduction)
0. [Usage](#usage)
0. [Notes](#notes)



## Introduction

This repository contains TensorFlow implementation code for the paper "L4: Practical loss-based stepsize adaptation for deep learning" (arxiv link). This work proposes an explicit rule for stepsize adaptation on top of existing optimizers such as Adam or momentum SGD.

*Disclaimer*: This code is a PROTOTYPE and most likely contains bugs. It should work with most Tensorflow models but most likely it doesn't comply with TensorFlow production code standards. Use at your own risk.


## Usage

Simply drop the L4.py file to your project and use as follows:
```python
import L4

...
opt = L4.L4Adam(fraction=0.20)
opt.minimize(loss)
...
```

or

```python
import L4

...
opt = L4.L4Mom()  # default value fraciton=0.15 is used
grads_and_vars = opt.compute_gradients(loss)
...
# Gradient manipulation
...

opt.apply_gradients(grads_and_vars, loss) # (!) Note that loss is passed here (!)
...
```

## Notes

*Contribute*: If you spot a bug or some incompatibility, contribute via a pull request! Thank you!
X