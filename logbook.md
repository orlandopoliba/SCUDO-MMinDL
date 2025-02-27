# Lesson 1
13/01/2025

Presentation of the course. Objectives, contents and motivations. Presentation of the exam modalities. Introduction. Deep learning in artificial intelligence. Examples of deep learning applications. History of industrial revolutions. The "AI Spring". Introduction to machine learning: task, experience, performance measure.

Slides for the introduction: [link](slides/Introduction.pdf)

# Lesson 2 
15/01/2025

The machine learning framework. Expected risk and empirical risk. Minimization of the empirical risk on the dataset. Linear regression. Using the `pandas` library in Python for data manipulation. Exploring the real estate dataset.

Notes for the machine learning framework: [link](notes/01%20-%20The%20Machine%20Learning%20Framework.pdf)

Python notebook on linear regression: [link](notebooks/01-linear_regression_house_prices.ipynb)

# Lesson 3 
20/01/2025

Linear regression. Explicit derivation of the optimal weights for linear regression. Train and test sets. Coefficient of determination. 

Notes for linear regression: [link](notes/02%20-%20Linear%20regression.pdf)

Python notebook on linear regression: [link](notebooks/01-linear_regression_house_prices.ipynb)

# Lesson 4
22/01/2025

The MNIST dataset. Introduction to binary classification. The logistic regression model. Sigmoid function. Entropy. Cross-entropy. Kullback-Leibler divergence. Jensen's inequality. Cross-entropy loss in binary logistic regression.

Notes for logistic regression: [link](notes/03%20-%20Logistic%20regression%20&%20Cross-entropy.pdf)

Python notebook on MNIST dataset: [link](notebooks/02-using_MNIST.ipynb)

# Lesson 5 
27/01/2025

Cross-entropy loss in binary logistic regression. Optimization of the cross-entropy loss. One-hot encoding of classes. Multi-class logistic regression. Softmax function. Cross-entropy loss in multi-class logistic regression.

Python notebook on binary logistic regression: [link](notebooks/03-binary_logistic_regression.ipynb)

# Lesson 6
29/01/2025

Multi-class logistic regression for the MNIST dataset. 

Likelihood of data in a parametric model. Equivalence between minimization of cross-entropy and of negative log-likelihood. 

Optimization algorithms. Geometrical meaning of the gradient. Gradient descent algorithm. Continuous counterpart: gradient flow. Decrease of the loss function along a gradient flow. Comments on the choice of the learning rate with examples.

Notes on logistic regression: [link](notes/03%20-%20Logistic%20regression%20&%20Cross-entropy.pdf)

Notes on Maximum Likelihood Estimators: [link](notes/04%20-%20Maximum%20likelihood%20estimators.pdf)

Notes on Gradient Descent: [link](notes/05%20-%20Gradient%20Descent.pdf)

Python notebook on multi-class logistic regression: [link](notebooks/04-multiclass_logistic_regression.ipynb)

Python notebook on gradient descent: [link](notebooks/05-gradient_descent.ipynb)

About the question "Does a minimum of the cross-entropy in logistic regression always exist?": [link](notes/04bis%20-%20About%20the%20existence%20of%20min%20for%20cross-entropy.pdf)

# Lesson 7
03/02/2025

Unrolling of the gradient descent algorithm for quadratic losses. Relationship between the learning rate and the eigenvalues of the Hessian matrix. Convergence of the gradient descent algorithm under suitable conditions on the loss function. The problem of local minima.

Introduction to neural networks. Interpreting linear regression as a one-layer neural network. Layers in a linear neural network. Expressiveness of neural networks. Activation functions. 

The perceptron as an artificial neuron. Expressing the NAND logical gate with a perceptron. Functional completeness of the NAND gate. 

Notes on Gradient Descent: [link](notes/05%20-%20Gradient%20Descent.pdf)

Python notebook on gradient descent: [link](notebooks/05-gradient_descent.ipynb)

Notes on neural networks: [link](notes/06%20-%20Neural%20Networks.pdf)

Notes on the perceptron: [link](notes/07%20-%20Perceptron.pdf)

# Lesson 8 
05/02/2025

Python classes. Definition of a class. Attributes and methods. The `__init__` method. The `self` keyword. Inheritance.

Creating a Python library. Folder structure. Importing from a module. Implementation of:
- the `Sequential` class for the definition of a multi-layered neural network 
- the `Layer` class for the definition of a layer in a neural network
- the `Linear` class for the definition of a linear layer in a neural network 
- the `Sigmoid` class for the definition of a sigmoid activation function
- the `forward` method in a layer and in a neural network
- the `Loss` class for the definition of a loss function
- the `MSE` class for the definition of the mean squared error loss function

The backpropagation algorithm. The chain rule. The backpropagation algorithm for a neural network. Implementation of:
- the `backward` method in the `MSE` class
- the `backward` method in the `Linear` class (for the computation of the gradient of the loss with respect to the weights)

Python notebook on Python classes: [link](notebooks/06-python_classes.ipynb)

Python notebook on building a neural network library: [link](notebooks/07-building_a_library.ipynb)

Python notebook with tests for the library: [link](notebooks/08-testing_the_library.ipynb)

Notes on backpropagation: [link](notes/08%20-%20Backpropagation.pdf)

Computation of the differential of the mean squared error loss function: [link](notes/09%20-%20Grads%20-%20MSE.pdf)

Computation of gradients in a linear layer: [link](notes/10%20-%20Grads%20-%20Linear%20layer.pdf)

Folder of the Python library: [link](mydl/)

# Lesson 9
10/02/2025

Implementation of:
- the `backward` method in the `Linear` class (for the computation of the gradient of the loss with respect to the biases and the input)
- the `backward` method in the `Sigmoid` class
- the `backward` method in the `Sequential` class (backpropagation through the layers of the network)
- the `Optimizer` class for the definition of an optimizer
- the `GD` class for the gradient descent optimizer
- the `train` method in the `Sequential` class

Using the library for a regression problem (Seoul Bike Sharing Demand dataset).

Computation of gradients in a linear layer: [link](notes/10%20-%20Grads%20-%20Linear%20layer.pdf)

Computations of gradients in nonlinear activation layers: [link](notes/11%20-%20Grads%20-%20Nonlinear%20activations.pdf)

Python notebook on building a neural network library: [link](notebooks/07-building_a_library.ipynb)

Python notebook with tests for the library: [link](notebooks/08-testing_the_library.ipynb)

Python notebook on regression (Seoul Bike Sharing Demand dataset): [link](notebooks/09-regression_bike_data.ipynb)

Folder of the Python library: [link](mydl/)

# Lesson 10
12/02/2025

Stochastic gradient descent. Mini-batch gradient descent. Implementation of batch training in the `train` method of the `Sequential` class.

Implementation of the `CrossEntropy` loss class. Testing the library on the MNIST dataset.

The Universal Approximation Theorem. 

Computation of differential of the cross-entropy loss function: [link](notes/12%20-%20Grads%20-%20Softmax%20and%20Cross-entropy.pdf)

Notes on stochastic gradient descent and mini-batch stochastic gradient descent: [link](notes/13%20-%20Stochastic%20Gradient%20Descent.pdf)

Notes on the Universal Approximation Theorem: [link](notes/14%20-%20Universal%20Appproximation%20Theorem.pdf)

Python notebook on building a neural network library: [link](notebooks/07-building_a_library.ipynb)

Python notebook with tests on MNIST dataset: [link](notebooks/10-mydl_on_MNIST.ipynb)

Python notebook on the Universal Approximation Theorem: [link](notebooks/11-universal_approximation_theorem.ipynb)

Folder of the Python library: [link](mydl/)