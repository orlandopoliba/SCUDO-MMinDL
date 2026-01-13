# SCUDO Course "Mathematical Methods in Deep Learning"
Repository with material for the SCUDO course "Mathematical Methods in Deep Learning" taught at Politecnico di Bari in 2025.

See the [logbook](logbook.md) for the detailed program of the course.

See the [references](references.md) for the list of references used in the course.

See the [list of topics not covered in the course](exam_topics.md) for a list of examples of topics that could be chosen for the exam, in case the student is interested in working on a topic not covered in the course (After all, we demand machines to perform on data they have never seen before, so why not ask the same to ourselves?). 

## Program
The objective of the course is to provide basic mathematical concepts and tools needed to have a clear insight into the mechanisms of neural networks and deep learning. Both theoretical and practical aspects will be covered, with hands-on exercise sessions for which the active participation of the students will be strongly encouraged.
 
The course will start with a general introduction to the objectives of data-driven machine learning and by explaining where neural networks and deep learning fit in this general framework. Then we will move on to the description of the basic building blocks of neural networks. We will introduce the concept of perceptron and of sigmoid neuron, and we will explain how these simple models can be used to build complex neural networks, with a gentle introduction to loss functions and the backpropagation algorithm. This will allow us to build a simple neural network that recognizes handwritten digits from the MNIST dataset with the aid of the PyTorch library from Python. With this concrete example in mind, we will discuss in detail some of the most important key aspects of neural networks.
 
We shall give an idea of the proof of the Universal Approximation Theorem, explaining how neural networks are able to approximate any continuous function. The proof will highlight the importance of the non-linearity in the activation functions of neural networks. Then we will refer to some results in the literature that show the benefits of using deep neural networks over shallow ones.
 
Next, we will move on to the choice of the loss function, discussing the choice of cross-entropy and log-likelihood loss functions. For this, we will recall some basic concepts of information theory and maximum likelihood estimation in statistics.
 
After the choice of the loss function, we will discuss the optimization of weights in neural networks. We will introduce, in a general setting, the algorithm of gradient descent, explaining how it is used to minimize functions. We will discuss pros and cons of the basic gradient descent algorithm. Then we will introduce in detail the algorithm used in practice to optimize the weights of neural networks, the stochastic gradient descent algorithm. Finally, we will discuss some of the most important variants of this algorithm, such as momentum update.
 
With these tools in hand, we will build a simple neural network from scratch.
 
We will move on to regularization techniques in neural networks, discussing the concepts of regularization and dropout. Moreover, we will discuss the problem of vanishing and exploding gradients in deep neural networks, and we will introduce the concept of batch normalization, explaining how it can be used to mitigate these problems.
 
If time permits, we will discuss some further topics in deep learning, such as convolutional neural networks for image recognition and natural language processing. A lecture will be dedicated to the question of how neural networks represent features in the data, discussing the concept of superposition of features and the Johnson-Lindenstrauss lemma.
 