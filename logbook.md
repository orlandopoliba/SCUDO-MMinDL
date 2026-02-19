# Lesson 1
27/01/2026

Objectives, contents and motivations. Presentation of the exam modalities. 

Deep learning in artificial intelligence. Examples of deep learning applications. History of industrial revolutions. The "AI Spring". 

Introduction to machine learning: task, experience, performance measure. Expected risk.

- [Introduction slides](slides/Introduction.pdf)

- [Notes on the Machine Learning Framework](notes/01%20-%20The%20Machine%20Learning%20Framework.pdf)

# Lesson 2 
29/01/2026

Empirical risk. Minimization of the empirical risk on the dataset. 

Linear regression. Using the `pandas` library in Python for data manipulation. Exploring the real estate dataset.

Linear regression as empirical risk minimization. Explicit derivation of the optimal weights for linear regression (normal equations). 

- [Personal notes on Linear Regression](notes/02%20-%20Linear%20regression.pdf)

- [Interactive notebook on empirical risk](notebooks/01-risk.py)

- [Marimo Python notebook on linear regression](notebooks/02-linear_regression.py)

- [Kaggle dataset on house prices](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

# Lesson 3 
03/02/2026

Train and test sets. Coefficient of determination. Application to the house prices dataset.

The MNIST dataset of handwritten digits.

Introduction to binary classification. The logistic regression model. Sigmoid function. Entropy. Cross-entropy. Kullback-Leibler divergence. Jensen's inequality.

- [Personal notes on Logistic Regression](notes/03%20-%20Logistic%20regression%20&%20Cross-entropy.pdf)

- [Marimo Python notebook on linear regression](notebooks/02-linear_regression.py)

- [Marimo Python notebook on using the MNIST dataset](notebooks/03-using_MNIST.py)

- [Interactive notebook on Kullback-Leibler divergence](notebooks/04-KL.py)

- [Kaggle dataset on house prices](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)


# Lesson 4
05/02/2026

Cross-entropy loss in binary logistic regression. Optimization of the cross-entropy loss. One-hot encoding of classes. Multi-class logistic regression. Softmax function. Cross-entropy loss in multi-class logistic regression.

Multi-class logistic regression for the MNIST dataset. 

- [Personal notes on Logistic Regression](notes/03%20-%20Logistic%20regression%20&%20Cross-entropy.pdf)

- [Marimo Python notebook on binary logistic regression](notebooks/05-binary_logistic_regression.py)

- [Marimo Python notebook on multiclass logistic regression](notebooks/06-multiclass_logistic_regression.py)

# Lesson 5 
09/02/2026

Optimization algorithms. Geometrical meaning of the gradient. Gradient descent algorithm. Continuous counterpart: gradient flow. Decrease of the loss function along a gradient flow. Comments on the choice of the learning rate with examples. 

Unrolling of the gradient descent algorithm for quadratic losses. Relationship between the learning rate and the eigenvalues of the Hessian matrix. Convergence of the gradient descent algorithm under suitable conditions on the loss function. The problem of local minima.

- [Personal notes on Gradient Descent](notes/04%20-%20Gradient%20Descent.pdf)

- [Interactive notebook on gradient descent](notebooks/07-gradient_descent.py)

# Lesson 6
10/02/2026

Introduction to neural networks. Interpreting linear regression as a one-layer neural network. Layers in a linear neural network. Expressiveness of neural networks. Activation functions. 

The perceptron as an artificial neuron. Expressing the NAND logical gate with a perceptron. Functional completeness of the NAND gate. 

Python classes. Definition of a class. Attributes and methods. The `__init__` method. The `self` keyword. Inheritance.

Creating a Python library. Folder structure. Importing from a module. Implementation of:
- the `Sequential` class for the definition of a multi-layered neural network 
- the `Layer` class for the definition of a layer in a neural network
- the `Linear` class for the definition of a linear layer in a neural network 
- the `Sigmoid` class for the definition of a sigmoid activation function
- the `forward` method in a layer and in a neural network
- the `Loss` class for the definition of a loss function
- the `MSE` class for the definition of the mean squared error loss function

- [Personal notes on neural networks](notes/05%20-%20Neural%20Networks.pdf)

- [Personal notes on the perceptron](notes/06%20-%20Perceptron.pdf)

- [Marimo Python notebook on Python classes](notebooks/08-python_classes.py)

Marimo notebooks with Python modules:
- [architecture module](notebooks/mydl/architecture.py)
- [layers module](notebooks/mydl/layers.py)
- [losses module](notebooks/mydl/losses.py)

# Lesson 7
17/02/2026

The backpropagation algorithm. The chain rule. The backpropagation algorithm in a multi-layered neural network. 

Explicit computation of backpropagation in the MSE loss function, in the linear layer and in the sigmoid activation function.

Implementation of:
- the `backward` method in the `MSE` class
- the `backward` method in the `Linear` class 
- the `backward` method in the `Sigmoid` class

- [Personal notes on backpropagation](notes/07%20-%20Backpropagation.pdf)

- [Personal notes on computing the backward pass in the MSE loss](notes/08%20-%20Grads%20-%20MSE.pdf)

- [Personal notes on computing the backward pass in the linear layer](notes/09%20-%20Grads%20-%20Linear%20layer.pdf)

- [Personal notes on computing the backward pass in the sigmoid activation function](notes/10%20-%20Grads%20-%20Nonlinear%20activations.pdf)

- [architecture module](notebooks/mydl/architecture.py)
- [layers module](notebooks/mydl/layers.py)
- [losses module](notebooks/mydl/losses.py)

# Lesson 8 
19/02/2026

Continuing the implementation of the library:
- the `backward` method in the `Sequential` class (backpropagation through the layers of the network)
- the `Optimizer` class for the definition of an optimizer
- the `GD` class for the gradient descent optimizer
- the `train` method in the `Sequential` class
  
Using the library for a regression problem (Seoul Bike Sharing Demand dataset).

- the `CrossEntropy` loss
- the `backward` method in the `CrossEntropy` loss

Using the library for a the MNIST dataset with a multi-layered neural network.

Graphical illustration of the Universal Approximation Theorem.

- [architecture module](notebooks/mydl/architecture.py)
- [optimizers module](notebooks/mydl/optimizers.py)
- [losses module](notebooks/mydl/losses.py)

- [Marimo Python notebook on using the Seoul Bike Sharing Demand dataset](notebooks/09-regression_bike_data.py)
- [Marimo Python notebook on using the MNIST dataset with a multi-layered neural network](notebooks/10-mydl_on_MNIST.py)
- [Interactive notebook on the Universal Approximation Theorem](notebooks/11-universal_approximation_theorem.py)