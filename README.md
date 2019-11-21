# Project Title

MNIST classifier using Tensorflow 2.0 and keras

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Please install Tensorflow 2.0 (obviously) and Numpy.


## Training

For training the model using a simple DNN:
```
python train_1.py
```

For training the model using a simple Inception Module:
```
python train_2.py
```

And for visualizing the training you could use tensorboard by executing:
```
tensorboard --logdir=./logs/scalars
```


## Built With

* [Tensorflow 2.0](https://www.tensorflow.org/) - Machine learning library 
* [Numpy 1.16.5](https://numpy.org/) - Math library


## Authors

* **Miguel Angel Campos** - *Software Engineer* - [Portfolio](http://mcampos.herokuapp.com/)