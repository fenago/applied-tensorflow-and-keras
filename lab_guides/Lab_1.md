
1. Introduction to Neural Networks and Deep Learning {#_idParaDest-18}
====================================================

::: {#_idContainer009 .Content}
:::

::: {#_idContainer024 .Content}
Overview

In this chapter, we will cover the basics of neural networks and how to
set up a deep learning programming environment. We will also explore the
common components and essential operations of a neural network . We will
conclude this chapter with an exploration of a trained neural network
created using TensorFlow. By the end of this chapter, you will be able
to train a neural network.


Introduction {#_idParaDest-19}
============

::: {#_idContainer024 .Content}
This chapter is about understanding what neural networks can do rather
than the finer workings of deep learning. For this reason, we will not
cover the mathematical concepts underlying deep learning algorithms but
will describe the essential pieces that make up a deep learning system
and the role of neural networks within that system. We will also look at
examples where neural networks have been used to solve real-world
problems using these algorithms.

At its core, this chapter challenges you to think about your problem as
a mathematical representation of ideas. By the end of this chapter, you
will be able to think about a problem as a collection of these
representations and to recognize how these representations can be
learned by deep learning algorithms.


What are Neural Networks? {#_idParaDest-20}
=========================

::: {#_idContainer024 .Content}
A **neural network** is a network of neurons. In our brain, we have a
network of billions of neurons that are interconnected with each other.
The neuron is one of the basic elements of the nervous system. The
primary function of the neuron is to perform actions as a response to an
event and transmit messages to other neurons. In this case, the action
is simply either activating or deactivating itself. Taking inspiration
from the brain\'s design, **artificial neural networks** were first
proposed in the 1940s by MIT professors *Warren McCullough* and *Walter
Pitts*.

Note

For more information on neural networks, refer to *Explained: Neural
networks. MIT News Office, April 14, 2017*, available at
<http://news.mit.edu/2017/explained-neural-networks-deep-learning-0414>.

Inspired by advancements in neuroscience, they proposed to create a
computer system that reproduced how the brain works (human or
otherwise). At its core was the idea of a computer system that worked as
an interconnected network, that is, a system that has many simple
components. These components interpret data and influence each other on
how to interpret that data. The same core idea r emains today.

Deep learning is largely considered the contemporary study of neural
networks. Think of it as a contemporary name given to neural networks.
The main difference is that the neural networks used in deep learning
are typically far greater in size, meaning they have many more nodes and
layers than earlier neural networks. Deep learning algorithms and
applications typically require resources to achieve success, hence the
use of the word *deep* to emphasize their size and the large number of
interconnected components.

[]{#_idTextAnchor021}

Successful Applications of Neural Networks {#_idParaDest-21}
------------------------------------------

Neural networks have been under research in one form or another since
their inception in the 1940s. It is only recently that deep learning
systems have been used successfully in large-scale industry
applications.

Contemporary proponents of neural networks have demonstrated great
success in speech recognition, language translation, image
classification, and other fields. Its current prominence is backed by a
significant increase in available computing power and the emergence of
**Graphic Processing Units** (**GPUs**) and **Tensor Processing Units**
(**TPUs**), which can perform many more simultaneous mathematical
operations than regular CPUs, as well as much greater availability of
data. Compared to CPUs, GPUs are designed to execute special tasks (in
the \"single instruction, multiple threads\" model) where the execution
can be parallelized.

One such success story is the power consumption of different AlphaGo
algorithms. **AlphaGo** is an initiative by DeepMind to develop a series
of algorithms to beat the game Go. It is considered a prime example of
the power of deep learning. The team at DeepMind was able to do this
using reinforcement learning in which AlphaGo becomes its own teacher.

The neural network, which initially knows nothing, plays with itself to
understand which moves lead to victory. The algorithm used TPUs for
training. TPUs are a type of chipset developed by Google that are
specialized for use in deep learning programs. The article *Alpha Zero:
Starting from scratch*,
<https://deepmind.com/blog/alphago-zero-learning-scratch/>, depicts the
number of GPUs and TPUs used to train different versions of the AlphaGo
algorithm.

Note

In this book, we will not be using GPUs to fulfill our activities. GPUs
are not required to work with neural networks. In several simple
examples---like the ones provided in this book---all computations can be
performed using a simple laptop\'s CPU. However, when dealing with very
large datasets, GPUs can be of great help given that the long time taken
to train a neural network would otherwise be impractical.

Here are a few examples where neural networks have had a significant
impact:

**Translating text**: In 2017, Google announced the release of a new
algorithm for its translation service called **Transformer**. The
algorithm consisted of a recurrent neural network called **Long
Short-term Memory** (**LSTM**) that was trained to use bilingual text.
LSTM is a form of neural network that is applied to text data. Google
showed that its algorithm had gained notable accuracy when compared to
the industry standard, **Bilingual Evaluation Understudy** (**BLEU**),
and was also computationally efficient. BLEU is an algorithm for
evaluating the performance of machine-translated text. For more
information on this, refer to the Google Research Blog, *Transformer: A
Novel Neural Network Architecture for Language Understanding,* August
31, 2017, available at
<https://research.googleblog.com/2017/08/transformer-novel-neural-network.html>.

**Self-driving vehicles**: Uber, NVIDIA, and Waymo are believed to be
using deep learning models to control different vehicle functions
related to driving. Each company is researching several possibilities,
including training the network using humans, simulating vehicles driving
in virtual environments, and even creating a small city-like environment
in which vehicles can be trained based on expected and unexpected
events.

Note

To know more about each of these achievements, refer to the following
references.

**Uber:** *Uber\'s new AI team is looking for the shortest route to
self-driving cars*, *Dave Gershgorn*, *Quartz*, *December 5, 2016*,
available at
<https://qz.com/853236/ubers-new-ai-team-is-looking-for-the-shortest-route-to-self-driving-cars/>.

**NVIDIA**: *End-to-End Deep Learning for Self-Driving Cars*, *August
17, 2016*, available at
<https://devblogs.nvidia.com/deep-learning-self-driving-cars/>.

**Waymo**: *Inside Waymo\'s Secret World for Training Self-Driving Cars.
The Atlantic*, *Alexis C. Madrigal*, *August 23, 2017*, available at
<https://www.theatlantic.com/technology/archive/2017/08/inside-waymos-secret-testing-and-simulation-facilities/537648/>.

**Image recognition**: Facebook and Google use deep learning models to
identify entities in images and automatically tag these entities as
persons from a set of contacts. In both cases, the networks are trained
with previously tagged images as well as with images from the target
friend or contact. Both companies report that the models can suggest a
friend or contact with a high level of accuracy in most cases.

While there are many more examples in other industries, the application
of deep learning models is still in its infancy. Many successful
applications are yet to come, including the ones that you create.


Why Do Neural Networks Work So Well? {#_idParaDest-22}
====================================

::: {#_idContainer024 .Content}
Why are neural networks so powerful? Neural networks are powerful
because they can be used to predict any given function with reasonable
approximation. If we can represent a problem as a mathematical function
and we have data that represents that function correctly, a deep
learning model can, given enough resources, be able to approximate that
function. This is typically called the *Universal Approximation
Theorem*. For more information, refer to Michael Nielsen: *Neural
Networks and Deep Learning: A visual proof that neural nets can compute
any function*, available at
<http://neuralnetworksanddeeplearning.com/chap4.html>.

We will not be exploring mathematical proofs of the universality
principle in this book. However, two characteristics of neural networks
should give you the right intuition on how to understand that principle:
representation learning and function approximation.

Note

For more information, refer to *A Brief Survey of Deep Reinforcement
Learning*, *Kai Arulkumaran, Marc Peter Deisenroth, Miles Brundage, and
Anil Anthony Bharath*, *arXiv*, *September 28, 2017*, available at
<https://www.arxiv-vanity.com/papers/1708.05866/>.

[]{#_idTextAnchor023}

Representation Learning {#_idParaDest-23}
-----------------------

The data used to train a neural network contains representations (also
known as *features*) that explain the problem you are trying to solve.
For instance, if we are interested in recognizing faces from images, the
color values of each pixel from a set of images that contain faces will
be used as a starting point. The model will then continuously learn
higher-level representations by combining pixels together as it goes
through its training process. A pictorial depiction is displayed here:

<div>

::: {#_idContainer010 .IMG---Figure}
![Figure 1.1: A series of higher-level representations based on input
data ](4_files/B15911_01_01.jpg)
:::

</div>

Figure 1.1: A series of higher-level representations based on input data

Note

*Figure 1.1* is a derivate image based on an original image from Yann
LeCun, Yoshua Bengio, and Geoffrey Hinton in *Deep Learning*, published
in *Nature, 521, 436--444 (28 May 2015) doi:10.1038/ nature14539*. You
can find the paper at: <https://www.nature.com/articles/nature14539>.

In formal words, neural networks are computation graphs in which each
step computes higher abstraction representations from input data. Each
of these steps represents a progression into a different abstraction
layer. Data progresses through each of these layers, thereby building
higher-level representations. The process finishes with the highest
representation possible: the one the model is trying to predict.

[]{#_idTextAnchor024}

Function Approximation {#_idParaDest-24}
----------------------

When neural networks learn new representations of data, they do so by
combining weights and biases with neurons from different layers. They
adjust the weights of these connections every time a training cycle
occurs using a mathematical technique called **backpropagation**. The
weights and biases improve at each round, up to the point that an
optimum is achieved. This means that a neural network can measure how
wrong it is on every training cycle, adjust the weights and biases of
each neuron, and try again. If it determines that a certain modification
produces better results than the previous round, it will invest in that
modification until an optimal solution is achieved.

So basically, in a single cycle, three things happen. The first one is
forward propagation where we calculate the results using weights,
biases, and inputs. In the second step, we calculate how far the
calculated value is from the expected value using a loss function. The
final step is to update the weights and biases moving in the reverse
direction of forward propagation, which is called backpropagation.

Since the weights and biases in the earlier layers do not have a direct
connection with the later layers, we use a mathematical tool called the
chain rule to calculate new weights for the earlier layers. Basically,
the change in the earlier layer is equal to the multiplication of the
gradients or derivatives of all the layers below it.

In a nutshell, that procedure is the reason why neural networks can
approximate functions. However, there are many reasons why a neural
network may not be able to predict a function with perfection, chief
among them being the following:

-   Many functions contain stochastic properties (that is, random
    properties).
-   There may be overfitting to peculiarities from the training data.
    Overfitting is a situation where the model we are training doesn\'t
    generalize well to data it has never seen before. It just learns the
    training data instead of finding some interesting patterns.
-   There may be a lack of training data.

In many practical applications, simple neural networks can approximate a
function with reasonable precision. These sorts of applications will be
our focus throughout this book.

[]{#_idTextAnchor025}

Limitations of Deep Learning {#_idParaDest-25}
----------------------------

Deep learning techniques are best suited to problems that can be defined
with formal mathematical rules (such as data representations). If a
problem is hard to define this way, it is likely that deep learning will
not provide a useful solution. Moreover, if the data available for a
given problem is either biased or only contains partial representations
of the underlying functions that generate that problem, deep learning
techniques will only be able to reproduce the problem and not learn to
solve it.

Remember that deep learning algorithms learn different representations
of data to approximate a given function. If data does not represent a
function appropriately, it is likely that the function will be
incorrectly represented by the neural network. Consider the following
analogy: you are trying to predict the national prices of gasoline (that
is, fuel) and create a deep learning model. You use your credit card
statement with your daily expenses on gasoline as an input data for that
model. The model may eventually learn the patterns of your gasoline
consumption, but it will likely misrepresent price fluctuations of
gasoline caused by other factors only represented weekly in your data
such as government policies, market competition, international politics,
and so on. The model will ultimately yield incorrect results when used
in production.

To avoid this problem, make sure that the data used to train a model
represents the problem the model is trying to address as accurately as
possible.

[]{#_idTextAnchor026}

Inherent Bias and Ethical Considerations {#_idParaDest-26}
----------------------------------------

Researchers have suggested that the use of deep learning models without
considering the inherent bias in the training data can lead not only to
poorly performing solutions but also to ethical complications.

For instance, in late 2016, researchers from the Shanghai Jiao Tong
University in China created a neural network that correctly classified
criminals using only pictures of their faces. The researchers used 1,856
images of Chinese men, of which half had been convicted. Their model
identified inmates with 89.5% accuracy.

Note

To know more about this, refer to
<https://blog.keras.io/the-limitations-of-deep-learning.html> and *MIT
Technology Review. Neural Network Learns to Identify Criminals by Their
Faces*, *November 22, 2016*, available at
<https://www.technologyreview.com/2016/11/22/107128/neural-network-learns-to-identify-criminals-by-their-faces/>.

The paper resulted in a great furor within the scientific community and
popular media. One key issue with the proposed solution is that it fails
to properly recognize the bias inherent in the input data. Namely, the
data used in this study came from two different sources: one for
criminals and one for non-criminals. Some researchers suggest that their
algorithm identifies patterns associated with the different data sources
used in the study instead of identifying relevant patterns from
people\'s faces. While there are technical considerations that we can
make about the reliability of the model, the key criticism is on ethical
grounds: researchers ought to clearly recognize the inherent bias in
input data used by deep learning algorithms and consider how its
application will impact on people\'s lives. *Timothy Revell. Concerns as
face recognition tech used to \'identify\' criminals. New Scientist.
December 1, 2016. Available at:*
[https://www.newscientist.com/article/2114900-concerns-as-face-recognition-tech-used-to-identify-criminals/](https://www.newscientist.com/article/2114900-concerns-as-face-recognition-tech-used-to-identify-crim).

There are different types of biases that can occur in a dataset.
Consider a case where you are building an automatic surveillance system
that can operate both in the daytime and nighttime. So, if your dataset
just included images from the daytime, then you would be introducing a
sample bias in the model. This could be eliminated by including
nighttime data and covering all the different types of cases possible,
such as images from a sunny day, a rainy day, and so on. Another example
to consider is where, let\'s suppose, a similar kind of system is
installed in a workplace to analyze the workers and their activities.
Now, if your model has been fed with thousands of examples with men
coding and women cooking, then this data clearly reflects stereotypes. A
solution to this problem is the same as earlier: to expose the model to
data that is more evenly distributed.

Note

To find out more about the topic of ethics in learning algorithms
(including deep learning), refer to the work done by the AI Now
Institute (<https://ainowinstitute.org/>), an organization created for
the understanding of the social implications of intelligent systems.

[]{#_idTextAnchor027}

Common Components and Operations of Neural Networks {#_idParaDest-27}
---------------------------------------------------

Neural networks have two key components: layers and nodes.

Nodes are responsible for specific operations, and layers are groups of
nodes that differentiate different stages of the system. Typically,
neural networks are comprised of the following three layers:

-   **Input layer**: Where the input data is received and interpreted
-   **Hidden layer**: Where computations take place, modifying the data
    as it passes through
-   **Output layer**: Where the output is assembled and evaluated

The following figure displays the working of layers of neural networks:

<div>

::: {#_idContainer011 .IMG---Figure}
![Figure 1.2: An illustration of the most common layers in a neural
network ](4_files/B15911_01_02.jpg)
:::

</div>

Figure 1.2: An illustration of the most common layers in a neural
network

Hidden layers are the most important layers in neural networks. They are
referred to as *hidden* because the representations generated in them
are not available in the data, but are learned from it instead. It is
within these layers where the main computations take place in neural
networks.

Nodes are where data is represented in the network. There are two values
associated with nodes: biases and weights. Both values affect how data
is represented by the nodes and passed on to other nodes. When a network
*learns*, it effectively adjusts these values to satisfy an optimization
function.

Most of the work in neural networks happens in the hidden layers.
Unfortunately, there isn\'t a clear rule for determining how many layers
or nodes a network should have. When implementing a neural network, you
will probably spend time experimenting with different combinations of
layers and nodes. It is advisable to always start with a single layer
and also with a number of nodes that reflect the number of features the
input data has (that is, how many *columns* are available in a given
dataset).

You can continue to add layers and nodes until a satisfactory
performance is achieved---or whenever the network starts overfitting to
the training data. Also, note that this depends very much on the dataset
-- if you were training a model to recognize hand-drawn digits, then a
neural network with two hidden layers would be enough, but if your
dataset was more complex, say for detecting objects like cars and
ambulance in images, then even 10 layers would not be enough and you
would need to have a deeper network for the objects to be recognized
correctly.

Likewise, if you were using a network with 100 hidden layers for
training on handwritten digits, then there would be a strong possibility
that you would overfit the model, as that much complexity would not be
required by the model in the first place.

Contemporary neural network practice is generally restricted to
experimentation with the number of nodes and layers (for example, how
deep the network is), and the kinds of operations performed at each
layer. There are many successful instances in which neural networks
outperformed other algorithms simply by adjusting these parameters.

To start off with, think about data entering a neural network system via
the input layer, and then moving through the network from node to node.
The path that data takes will depend on how interconnected the nodes
are, the weights and the biases of each node, the kind of operations
that are performed in each layer, and the state of the data at the end
of such operations. Neural networks often require many **runs** (or
epochs) in order to keep tuning the weights and biases of nodes, meaning
that data flows over the different layers of the graph multiple times.


Configuring a Deep Learning Environment {#_idParaDest-28}
=======================================

::: {#_idContainer024 .Content}
Before we finish this chapter, we want you to interact with a real
neural network. We will start by covering the main software components
used throughout this book and make sure that they are properly
installed. We will then explore a pre-trained neural network and explore
a few of the components and operations discussed in the *What are Neural
Networks?* section.

[]{#_idTextAnchor029}

Software Components for Deep Learning {#_idParaDest-29}
-------------------------------------

We\'ll use the following software components for deep learning:

[]{#_idTextAnchor030}

### Python 3 {#_idParaDest-30}

We will be using Python 3 in this book. Python is a general-purpose
programming language that is very popular with the scientific
community---hence its adoption in deep learning. Python 2 is not
supported in this book but can be used to train neural networks instead
of Python 3. Even if you chose to implement your solutions in Python 2,
consider moving to Python 3 as its modern feature set is far more robust
than that of its predecessor.

[]{#_idTextAnchor031}

### TensorFlow {#_idParaDest-31}

TensorFlow is a library used for performing mathematical operations in
the form of graphs. TensorFlow was originally developed by Google, and
today, it is an open source project with many contributors. It has been
designed with neural networks in mind and is among the most popular
choices when creating deep learning algorithms.

TensorFlow is also well known for its production components. It comes
with TensorFlow Serving (<https://github.com/tensorflow/serving>), a
high-performance system for serving deep learning models. Also, trained
TensorFlow models can be consumed in other high-performance programming
languages such as Java, Go, and C. This means that you can deploy these
models on anything from a micro-computer (that is, a Raspberry Pi) to an
Android device. As of November 2019, TensorFlow version 2.0 is the
latest version.

[]{#_idTextAnchor032}

### Keras {#_idParaDest-32}

In order to interact efficiently with TensorFlow, we will be using Keras
(<https://keras.io/>), a Python package with a high-level API for
developing neural networks. While TensorFlow focuses on components that
interact with each other in a computational graph, Keras focuses
specifically on neural networks. Keras uses TensorFlow as its backend
engine and makes developing such applications much easier.

As of November 2019, Keras is the built-in and default API of
TensorFlow. It is available under the `tf.keras`{.literal} namespace.

[]{#_idTextAnchor033}

### TensorBoard {#_idParaDest-33}

TensorBoard is a data visualization suite for exploring TensorFlow
models and is natively integrated with TensorFlow. TensorBoard works by
consuming the checkpoint and summary files created by TensorFlow as it
trains a neural network. Those can be explored either in near real time
(with a 30-second delay) or after the network has finished training.
TensorBoard makes the process of experimenting with and exploring a
neural network much easier---plus, it\'s quite exciting to follow the
training of your network.

[]{#_idTextAnchor034}

### Jupyter Notebook, Pandas, and NumPy {#_idParaDest-34}

When working to create deep learning models with Python, it is common to
start working interactively; slowly developing a model that eventually
turns into more structured software. Three Python packages are used
frequently during this process: Jupyter Notebooks, Pandas, and NumPy:

-   Jupyter Notebook create interactive Python sessions that use a web
    browser as their interface.
-   Pandas is a package for data manipulation and analysis.
-   NumPy is frequently used for shaping data and performing numerical
    computations.

These packages are used occasionally throughout this book. They
typically do not form part of a production system but are often used
when exploring data and starting to build a model. We\'ll focus on the
other tools in much more detail.

Note

The books *Learning pandas* by Michael Heydt (June 2017, Packt
Publishing), available at
<https://www.packtpub.com/big-data-and-business-intelligence/learning-pandas-second-edition>,
and *Learning Jupyter* by Dan Toomey (November 2016, Packt Publishing),
available at
<https://www.packtpub.com/big-data-and-business-intelligence/learning-jupyter-5-second-edition>,
both offer comprehensive guides on how to use these technologies. These
books are good references for continuing to learn more.

The following table details the software requirements required for
successfully creating the deep learning models explained in this book:

<div>

::: {#_idContainer012 .IMG---Figure}
![Figure 1.3: The software components necessary for creating a deep
learning environment ](5_files/B15911_01_03.jpg)
:::

</div>

Figure 1.3: The software components necessary for creating a deep
learning environment

Anaconda is a free distribution of many useful Python packages for
Windows, Mac or other platform. We recommend that you follow the
instructions at <https://docs.anaconda.com/anaconda/install/>. The
standard Anaconda installation will install most of these components and
the first exercise will work through how to install the others.

[]{#_idTextAnchor035}

Exercise 1.01: Verifying the Software Components {#_idParaDest-35}
------------------------------------------------

Before we explore a trained neural network, let\'s verify whether all
the software components that we need are available. We have included a
script that verifies whether these components work. Let\'s take a moment
to run the script and deal with any eventual problems we may find. We
will now be testing whether the software components required for this
book are available in your working environment. First, we suggest the
creation of a Python virtual environment using Python\'s native module
`venv`{.literal}. Virtual environments are used for managing project
dependencies. We suggest each project you create has its own virtual
environment.

1.  A python virtual environment can be created by using the following
    command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ python -m venv venv
    $ source venv/bin/activate
    ```
    :::

    The latter command will append the string `venv`{.literal} at the
    beginning of the command line.

    Make sure you always activate your Python virtual environment when
    working on a project. To deactivate your virtual environment, run
    `$ deactivate`{.literal}.

2.  After activating your virtual environment, make sure that the right
    components are installed by executing `pip`{.literal} over the
    `requirements.txt`{.literal} file (<https://packt.live/300skHu>).

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ pip install –r requirements.txt
    ```
    :::

    The output is as follows:

    ::: {#_idContainer013 .IMG---Figure}
    ![Figure 1.4: A screenshot of a Terminal running pip to install
    dependencies from requirements.txt ](5_files/B15911_01_04.jpg)
    :::

    Figure 1.4: A screenshot of a Terminal running pip to install
    dependencies from requirements.txt

3.  This will install the libraries used in this book in that virtual
    environment. It will do nothing if they are already available. If
    the library is getting installed, a progress bar will be shown, else
    it will notify that
    \'`requirement is already specified`{.literal}\'. To check the
    available libraries installed, please use the following command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ pip list
    ```
    :::

    The output will be as follows:

    ::: {#_idContainer014 .IMG---Figure}
    ![Figure 1.5: A screenshot of a Terminal running pip to list the
    available libraries ](5_files/B15911_01_05.jpg)
    :::

    Figure 1.5: A screenshot of a Terminal running pip to list the
    available libraries

    Note

    These libraries are essential for working with all the code
    activities in this book.

4.  As a final step in this exercise, execute the script
    `test_stack.py`{.literal}. This can be found at:
    <https://packt.live/2B0JNau> It verifies that all the required
    packages for this book are installed and available in your system.

5.  Run the following script to check if the dependencies of Python 3,
    TensorFlow, and Keras are available. Use the following command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ python3 Chapter01/Exercise1.01/test_stack.py
    ```
    :::

    The script returns helpful messages stating what is installed and
    what needs to be installed:

    ::: {#_idContainer015 .IMG---Figure}
    ![Figure 1.6: A screenshot of a Terminal displaying that not all the
    requirements are installed ](5_files/B15911_01_06.jpg)
    :::

    Figure 1.6: A screenshot of a Terminal displaying that not all the
    requirements are installed

    For example, in the preceding screenshot, it shows that TensorFlow
    2.0 is not detected but Keras 2.2 or higher is detected. Hence you
    are shown the error message
    `Please review software requirements before proceeding to Lesson 2`{.literal}.
    If all the requirements are fulfilled, then it will show Python,
    TensorFlow, and Keras as installed, as shown in the following
    screenshot:

    ::: {#_idContainer016 .IMG---Figure}
    ![Figure 1.7: A screenshot of the Terminal displaying that all
    elements are installed ](5_files/B15911_01_07.jpg)
    :::

    Figure 1.7: A screenshot of the Terminal displaying that all
    elements are installed

6.  Run the following script command in your Terminal to find more
    information on how to configure TensorBoard:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ tensorboard –help
    ```
    :::

    The output is as follows:

<div>

::: {#_idContainer017 .IMG---Figure}
![Figure 1.8: An output of the \--help command
](5_files/B15911_01_08.jpg)
:::

</div>

Figure 1.8: An output of the \--help command

You should see the relevant help messages that explain what each command
does, as in *Figure 1.8*.

As you can see in the figure above, the script returns messages
informing you that all dependencies are installed correctly.

Note

To access the source code for this specific section, please refer to
<https://packt.live/2B0JNau>.

This section does not currently have an online interactive example, and
will need to be run locally.

Once we have verified that Python 3, TensorFlow, Keras, TensorBoard, and
the packages outlined in `requirements.txt`{.literal} have been
installed, we can continue to a demo on how to train a neural network
and then go on to explore a trained network using these same tools.


Exploring a Trained Neural Network {#_idParaDest-36}
==================================

::: {#_idContainer024 .Content}
In this section, we\'ll explore a trained neural network. We\'ll do this
to understand how a neural network solves a real-world problem
(predicting handwritten digits) and to get familiar with the TensorFlow
API. When exploring this neural network, we will recognize many
components introduced in previous sections, such as nodes and layers,
but we will also see many that we don\'t recognize (such as activation
functions); we will explore those in further sections. We will then walk
through an exercise on how that neural network was trained and then
train that same network ourselves.

The network that we will be exploring has been trained to recognize
numerical digits (integers) using images of handwritten digits. It uses
the MNIST dataset (<http://yann.lecun.com/exdb/mnist/>), a classic
dataset frequently used for exploring pattern recognition tasks.

[]{#_idTextAnchor037}

The MNIST Dataset {#_idParaDest-37}
-----------------

The **Modified National Institute of Standards and Technology**
(**MNIST**) dataset contains a training set of 60,000 images and a test
set of 10,000 images. Each image contains a single handwritten number.
This dataset, which is derived from one created by the US Government,
was originally used to test different approaches to the problem of
recognizing handwritten text by computer systems. Being able to do that
was important for the purpose of increasing the performance of postal
services, taxation systems, and government services. The MNIST dataset
is considered too naïve for contemporary methods. Different and more
recent datasets are used in contemporary research (for example,
**Canadian Institute for Advanced Research** (**CIFAR**). However, the
MNIST dataset is still very useful for understanding how neural networks
work because known models can achieve a high level of accuracy with
great efficiency.

Note

The CIFAR dataset is a machine learning dataset that contains images
organized in different classes. Different than the MNIST dataset, the
CIFAR dataset contains classes from many different areas including
animals, activities, and objects. The CIFAR dataset is available at
<https://www.cs.toronto.edu/~kriz/cifar.html>.

However, the MNIST dataset is still very useful for understanding how
neural networks work because known models can achieve a high level of
accuracy with great efficiency. In the following figure, each image is a
separate 20x20-pixel image containing a single handwritten digit. You
can find the original dataset at <http://yann.lecun.com/exdb/mnist/>.

<div>

::: {#_idContainer018 .IMG---Figure}
![Figure 1.9: An excerpt from the training set of the MNIST dataset
](6_files/B15911_01_09.jpg)
:::

</div>

Figure 1.9: An excerpt from the training set of the MNIST dataset

[]{#_idTextAnchor038}

Training a Neural Network with TensorFlow {#_idParaDest-38}
-----------------------------------------

Now, let\'s train a neural network to recognize new digits using the
MNIST dataset. We will be implementing a special-purpose neural network
called a **Convolutional Neural Network**(**CNN**) to solve this problem
(we will discuss those in more detail in later sections). Our complete
network contains three hidden layers: two fully connected layers and a
convolutional layer. The model is defined by the following TensorFlow
snippet of Python code:

Note

The code snippet shown here uses a backslash ( `\`{.literal} ) to split
the logic across multiple lines. When the code is executed, Python will
ignore the backslash, and treat the code on the next line as a direct
continuation of the current line.

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model = Sequential()
model.add(Convolution2D(filters = 10, kernel_size = 3, \
                        input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))
```
:::

Note

Use the `mnist.py`{.literal} file for your reference at
<https://packt.live/2Cuhj9w>. Follow along by opening the script in your
code editor.

We execute the preceding snippet of code only once during the training
of our network.

We will go into a lot more detail about each one of those components
using Keras in *Chapter 2*, *Real-World Deep Learning with TensorFlow
and Keras: Predicting the Price of Bitcoin*. For now, we\'ll focus on
understanding that the network is altering the values of the
`Weights`{.literal} and `Biases`{.literal} in each layer on every run.
These lines of Python are the culmination of dozens of years of neural
network research.

Now let\'s train that network to evaluate how it performs in the MNIST
dataset.

[]{#_idTextAnchor039}

Exercise 1.02: Training a Neural Network Using the MNIST Dataset {#_idParaDest-39}
----------------------------------------------------------------

In this exercise, we will train a neural network for detecting
handwritten digits from the MNIST dataset. Execute the following steps
to set up this exercise:

1.  Open two Terminal instances.

2.  Navigate to <https://packt.live/2BWNAWK>. Ensure that your Python 3
    virtual environment is active and that the requirements outlined in
    `requirements.txt`{.literal} are installed.

3.  In one of them, start a TensorBoard server with the following
    command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ tensorboard --logdir logs/fit
    ```
    :::

    The output is as follows:

    ::: {#_idContainer019 .IMG---Figure}
    ![Figure 1.10: The TensorBoard server ](6_files/B15911_01_10.jpg)
    :::

    Figure 1.10: The TensorBoard server

    In the other, run the `mnist.py`{.literal} script from within that
    directory with the following command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ python mnist.py
    ```
    :::

    When you start running the script, you will see the progress bar as
    follows:

    ::: {#_idContainer020 .IMG---Figure}
    ![Figure 1.11: The result of the mnist.py script
    ](6_files/B15911_01_11.jpg)
    :::

    Figure 1.11: The result of the mnist.py script

4.  Open your browser and navigate to the TensorBoard URL provided when
    you start the server in *step 3*, it might be
    `http://localhost:6006/`{.literal} or similar. In the Terminal where
    you ran the `mnist.py`{.literal} script, you will see a progress bar
    with the epochs of the model. When you open the browser page, you
    will see a couple of graphs, `epoch_accuracy`{.literal} and
    `epoch_loss`{.literal} graphs. Ideally, the accuracy should improve
    with each iteration and the loss should decrease with each
    iteration. You can confirm this visually with the graphs.

5.  Click the `epoch_accuracy`{.literal} graph, enlarge it, and let the
    page refresh (or click on the `refresh`{.literal} icon). You will
    see the model gaining accuracy as the epochs go by:

<div>

::: {#_idContainer021 .IMG---Figure}
![Figure 1.12: A visualization of the accuracy and loss graphs using
TensorBoard ](6_files/B15911_01_12.jpg)
:::

</div>

Figure 1.12: A visualization of the accuracy and loss graphs using
TensorBoard

We can see that after about 5 epochs (or steps), the network surpassed
97% accuracy. That is, the network is getting 97% of the digits in the
test set correct by this point.

Note

To access the source code for this specific section, please refer to
<https://packt.live/2Cuhj9w>.

This section does not currently have an online interactive example, and
will need to be run locally.

Now, let\'s also test how well those networks perform with unseen data.

[]{#_idTextAnchor040}

Testing Network Performance with Unseen Data {#_idParaDest-40}
--------------------------------------------

Visit the website <http://mnist-demo.herokuapp.com/> in your browser and
draw a number between 0 and 9 in the designated white box:

<div>

::: {#_idContainer022 .IMG---Figure}
![Figure 1.13: A web application for manually drawing digits and testing
the accuracy of two trained networks ](6_files/B15911_01_13.jpg)
:::

</div>

Figure 1.13: A web application for manually drawing digits and testing
the accuracy of two trained networks

Note

This web application we are using was created by *Shafeen Tejani* to
explore whether a trained network can correctly predict handwritten
digits that we create.

Source: <https://github.com/ShafeenTejani/mnist-demo>.

In the application, you can see the results of two neural networks -- a
**Convolutional Neural Network** (**CNN**) and a **Fully Connected
Neural Network**. The one that we have trained is the CNN. Does it
classify all your handwritten digits correctly? Try drawing numbers at
the edge of the designated area. For instance, try drawing the number
`1`{.literal} close to the right edge of the drawing area, as shown in
the following figure:

<div>

::: {#_idContainer023 .IMG---Figure}
![Figure 1.14: Both networks have a difficult time estimating values
drawn on the edges of the area ](6_files/B15911_01_14.jpg)
:::

</div>

Figure 1.14: Both networks have a difficult time estimating values drawn
on the edges of the area

In this example, we see the number 1 drawn to the right side of the
drawing area. The probability of this number being a 1 is 0 in both
networks.

The MNIST dataset does not contain numbers on the edges of images.
Hence, neither network assigns relevant values to the pixels located in
that region. Both networks are much better at classifying numbers
correctly if we draw them closer to the center of the designated area.
This is due to the fact that in the training set, we only had images
with numbers drawn in the center of the image. This shows that neural
networks can only be as powerful as the data that is used to train them.
If the data used for training is very different than what we are trying
to predict, the network will most likely produce disappointing results.

[]{#_idTextAnchor041}

Activity 1.01: Training a Neural Network with Different Hyperparameters {#_idParaDest-41}
-----------------------------------------------------------------------

In this section, we will explore the neural network that we trained
during our work on *Exercise 1.02*, *Training a Neural Network Using the
MNIST Dataset*, where we trained our own CNN on the MNIST dataset. We
have provided that same trained network as binary files in the directory
of this book. In this activity, we will just cover the things that you
can do using TensorBoard and we will train several other networks by
just changing some hyperparameters.

Here are the steps you need to follow:

1.  Open TensorBoard by writing the appropriate command.

2.  Open the TensorBoard accuracy graph and play with the values of
    smoothening sliders in scalar areas.

3.  Train another model by changing the hyperparameters.

4.  Try decreasing the learning rate and increasing the number of
    epochs.

5.  Now try to understand what effect this hyperparameter tuning has on
    the graphs generated on TensorBoard.

6.  Try increasing the learning rate and decreasing the number of epochs
    and repeat *step 5*.

    Note:

    The solution for this activity can be found on page 130.


Summary {#_idParaDest-42}
=======

::: {#_idContainer024 .Content}
In this chapter, we explored a TensorFlow-trained neural network using
TensorBoard and trained our own modified version of that network with
different epochs and learning rates. This gave you hands-on experience
of how to train a highly performant neural network and allowed you to
explore some of its limitations.

Do you think we can achieve similar accuracy with real Bitcoin data? We
will attempt to predict future Bitcoin prices using a common neural
network algorithm in *Chapter 2*, *Real-World Deep Learning with
TensorFlow and Keras: Predicting the Price of Bitcoin*. In *Chapter 3*,
*Real-World Deep Learning with TensorFlow and Keras: Evaluating the
Bitcoin Model*, we will evaluate and improve that model, and finally, in
*Chapter 4*, *Productization*, we will create a program that serves the
prediction of that system via an HTTP API.
