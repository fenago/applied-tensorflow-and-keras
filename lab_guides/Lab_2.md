
2. Real-World Deep Learning: Predicting the Price of Bitcoin {#_idParaDest-43}
============================================================

::: {#_idContainer028 .Content}
:::

::: {#_idContainer057 .Content}
Overview

This chapter will help you to prepare data for a deep learning model,
choose the right model architecture, use Keras---the default API of
TensorFlow 2.0, and make predictions with the trained model. By the end
of this chapter, you will have prepared a model to make predictions
which we will explore in the upcoming chapters.


Introduction {#_idParaDest-44}
============

::: {#_idContainer057 .Content}
Building on fundamental concepts from *Chapter 1*, *Introduction to
Neural Networks and Deep Learning*, let\'s now move on to a real-world
scenario and identify whether we can build a deep learning model that
predicts Bitcoin prices.

We will learn the principles of preparing data for a deep learning
model, and how to choose the right model architecture. We will use
Keras---the default API of TensorFlow 2.0 and make predictions with the
trained model. We will conclude this chapter by putting all these
components together and building a bare bones, yet complete, first
version of a deep learning application.

Deep learning is a field that is undergoing intense research activity.
Among other things, researchers are devoted to inventing new neural
network architectures that can either tackle new problems or increase
the performance of previously implemented architectures.

In this chapter, we will study both old and new architectures. Older
architectures have been used to solve a large array of problems and are
generally considered the right choice when starting a new project. Newer
architectures have shown great success in specific problems but are
harder to generalize. The latter are interesting as references of what
to explore next but are hardly a good choice when starting a project.

The following topic discusses the details of these architectures and how
to determine the best one for a particular problem statement.


Choosing the Right Model Architecture {#_idParaDest-45}
=====================================

::: {#_idContainer057 .Content}
Considering the available architecture possibilities, there are two
popular architectures that are often used as starting points for several
applications: **Convolutional Neural Networks** (**CNNs**) and
**Recurrent Neural Networks** (**RNNs**). These are foundational
networks and should be considered starting points for most projects.

We also include descriptions of another three networks, due to their
relevance in the field: **Long Short-Term Memory** (**LSTM**) networks
(an RNN variant); **Generative Adversarial Networks** (**GANs**); and
**Deep Reinforcement Learning** (**DRL**). These latter architectures
have shown great success in solving contemporary problems, however, they
are slightly difficult to use. The next section will cover the use of
different types of architecture in different problems.

[]{#_idTextAnchor046}

Convolutional Neural Networks (CNNs) {#_idParaDest-46}
------------------------------------

CNNs have gained notoriety for working with problems that have a
grid-like structure. They were originally created to classify images,
but have been used in several other areas, ranging from speech
recognition to self-driving vehicles.

A CNN\'s essential insight is to use closely related data as an element
of the training process, instead of only individual data inputs. This
idea is particularly effective in the context of images, where a pixel
located at the center is related to the ones located to its right and
left. The name **convolution** is given to the mathematical
representation of the following process:

<div>

::: {#_idContainer029 .IMG---Figure}
![Figure 2.1: Illustration of the convolution process
](3_files/B15911_02_01.jpg)
:::

</div>

Figure 2.1: Illustration of the convolution process

Note

Image source: Volodymyr Mnih, *et al.*

You can find this image at: <https://packt.live/3fivWLB>

For more information about deep reinforcement learning, refer to
*Human-level Control through Deep Reinforcement Learning*. *February
2015*, *Nature*, available at
<https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>.

[]{#_idTextAnchor047}

Recurrent Neural Networks (RNNs) {#_idParaDest-47}
--------------------------------

A CNN works with a set of inputs that keeps altering the weights and
biases of the network\'s respective layers and nodes. A known limitation
of this approach is that its architecture ignores the sequence of these
inputs when determining the changes to the network\'s weights and
biases.

RNNs were created precisely to address that problem. They are designed
to work with sequential data. This means that at every epoch, layers can
be influenced by the output of previous layers. The memory of previous
observations in each sequence plays an important role in the evaluation
of posterior observations.

RNNs have had successful applications in speech recognition due to the
sequential nature of that problem. Also, they are used for translation
problems. Google Translate\'s current algorithm---Transformer, uses an
RNN to translate text from one language to another. In late 2018, Google
introduced another algorithm based on the Transformer algorithm called
**Bidirectional Encoder Representations from Transformers** (**BERT**),
which is currently state of the art in **Natural Language Processing**
(**NLP**).

Note

For more information RNN applications, refer to the following:

*Transformer: A Novel Neural Network Architecture for Language
Understanding*, *Jakob Uszkoreit*, *Google Research Blog*, *August
2017*, available at
<https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html>.

*BERT: Open Sourcing BERT: State-of-the-Art Pre-Training for Natural
Language Processing*, available at
<https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html>.

The following diagram illustrates how words in English are linked to
words in French, based on where they appear in a sentence. RNNs are very
popular in language translation problems:

<div>

::: {#_idContainer030 .IMG---Figure}
![Figure 2.2: Illustration from distill.pub linking words in English and
French ](3_files/B15911_02_02.jpg)
:::

</div>

Figure 2.2: Illustration from distill.pub linking words in English and
French

Note

Image source: <https://distill.pub/2016/augmented-rnns/>

[]{#_idTextAnchor048}

Long Short-Term Memory (LSTM) Networks {#_idParaDest-48}
--------------------------------------

**LSTM** networks are RNN variants created to address the vanishing
gradient problem. This problem is caused by memory components that are
too distant from the current step that receive lower weights due to
their distance. LSTMs are a variant of RNNs that contain a memory
component called a **forget gate**. This component can be used to
evaluate how both recent and old elements affect the weights and biases,
depending on where the observation is placed in a sequence.

Note

The LSTM architecture was first introduced by Sepp Hochreiter and Jürgen
Schmidhuber in 1997. Current implementations have had several
modifications. For a detailed mathematical explanation of how each
component of an LSTM works, refer to the article *Understanding LSTM
Networks*, by Christopher Olah, August 2015, available at
<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>.

[]{#_idTextAnchor049}

Generative Adversarial Networks {#_idParaDest-49}
-------------------------------

**Generative Adversarial Networks** (**GANs**) were invented in 2014 by
Ian Goodfellow and his colleagues at the University of Montreal. GANs
work based on the approach that, instead of having one neural network
that optimizes weights and biases with the objective to minimize its
errors, there should be two neural networks that compete against each
other for that purpose.

Note

For more information on GANs, refer to *Generative Adversarial
Networks*, *Ian Goodfellow, et al.*, *arXiv*. *June 10, 2014*, available
at <https://arxiv.org/pdf/1406.2661.pdf>.

GANs generate new data (*fake* data) and a network that evaluates the
likelihood of the data generated by the first network being *real* or
*fake*. They compete because both learn: one learns how to better
generate *fake* data, and the other learns how to distinguish whether
the data presented is real. They iterate on every epoch until
convergence. That is the point when the network that evaluates generated
data cannot distinguish between *fake* and *real* data any further.

GANs have been successfully used in fields where data has a clear
topological structure. Originally, GANs were used to create synthetic
images of objects, people\'s faces, and animals that were similar to
real images of those things. You will see in the following image, used
in the **StarGAN** project, that the expressions on the face change:

<div>

::: {#_idContainer031 .IMG---Figure}
![Figure 2.3: Changes in people\'s faces based on emotion, using GAN
algorithms ](3_files/B15911_02_03.jpg)
:::

</div>

Figure 2.3: Changes in people\'s faces based on emotion, using GAN
algorithms

This domain of image creation is where GANs are used the most
frequently, but applications in other domains occasionally appear in
research papers.

Note

Image source: StarGAN project, available at
<https://github.com/yunjey/StarGAN>.

[]{#_idTextAnchor050}

Deep Reinforcement Learning (DRL) {#_idParaDest-50}
---------------------------------

The original DRL architecture was championed by DeepMind, a Google-owned
artificial intelligence research organization based in the UK. The key
idea of DRL networks is that they are unsupervised in nature and that
they learn from trial and error, only optimizing for a reward function.

That is, unlike other networks (which use a supervised approach to
optimize incorrect predictions as compared to what are known to be
correct ones), DRL networks do not know of a correct way of approaching
a problem. They are simply given the rules of a system and are then
rewarded every time they perform a function correctly. This process,
which takes a very large number of iterations, eventually trains
networks to excel in several tasks.

Note

For more information about DRL, refer to *Human-Level Control through
Deep Reinforcement Learning*, *Volodymyr Mnih et al.*, *February 2015*,
*Nature*, available at:
<https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>.

DRL models gained popularity after DeepMind created AlphaGo---a system
that plays the game Go better than professional players. DeepMind also
created DRL networks that learn how to play video games at a superhuman
level, entirely on their own.

Note

For more information about DQN, look up the DQN that was created by
DeepMind to beat Atari games. The algorithm uses a DRL solution to
continuously increase its reward.

Image source: <https://keon.io/deep-q-learning/>.

Here\'s a summary of neural network architectures and their
applications:

<div>

::: {#_idContainer032 .IMG---Figure}
![Figure 2.4: Different neural network architectures, data structures,
and their successful applications ](3_files/B15911_02_04.jpg)
:::

</div>

Figure 2.4: Different neural network architectures, data structures, and
their successful applications

[]{#_idTextAnchor051}

Data Normalization {#_idParaDest-51}
------------------

Before building a deep learning model, data normalization is an
important step. Data normalization is a common practice in machine
learning systems. For neural networks, researchers have proposed that
normalization is an essential technique for training RNNs (and LSTMs),
mainly because it decreases the network\'s training time and increases
the network\'s overall performance.

Note

For more information, refer to *Batch Normalization: Accelerating Deep
Network Training by Reducing Internal Covariate Shift*, *Sergey Ioffe et
al.*, *arXiv*, March 2015, available at
<https://arxiv.org/abs/1502.03167>.

Which normalization technique works best depends on the data and the
problem at hand. A few commonly used techniques are listed here:

[]{#_idTextAnchor052}

### Z-Score {#_idParaDest-52}

When data is normally distributed (that is, Gaussian), you can compute
the distance between each observation as a standard deviation from its
mean. This normalization is useful when identifying how distant the data
points are from more likely occurrences in the distribution. The Z-score
is defined by the following formula:

<div>

::: {#_idContainer033 .IMG---Figure}
![Figure 2.5: Formula for Z-Score ](3_files/B15911_02_05.jpg)
:::

</div>

Figure 2.5: Formula for Z-Score

Here, *x*[i]{.subscript} is the *i*[th]{.superscript} observation,
![1](3_files/B15911_02_Formula_01.png) is the mean, and
![2](3_files/B15911_02_Formula_02.png) is the standard deviation of the
series.

Note

For more information, refer to the standard score article (*Z-Score:
Definition, Formula, and Calculation*), available at
<https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/z-score/>.

[]{#_idTextAnchor053}

### Point-Relative Normalization {#_idParaDest-53}

This normalization computes the difference in a given observation in
relation to the first observation of the series. This kind of
normalization is useful for identifying trends in relation to a starting
point. The point-relative normalization is defined by:

<div>

::: {#_idContainer036 .IMG---Figure}
![Figure 2.6: Formula for point-relative normalization
](3_files/B15911_02_06.jpg)
:::

</div>

Figure 2.6: Formula for point-relative normalization

Here, *o*[i]{.subscript} is the *i*[th]{.superscript} observation, and
*o*[o]{.subscript} is the first observation of the series.

Note

For more information on making predictions, watch *How to Predict Stock
Prices Easily -- Intro to Deep Learning \#7*, *Siraj Raval*, available
on YouTube at <https://www.youtube.com/watch?v=ftMq5ps503w>.

[]{#_idTextAnchor054}

### Maximum and Minimum Normalization {#_idParaDest-54}

This type of normalization computes the distance between a given
observation and the maximum and minimum values of the series. This is
useful when working with series in which the maximum and minimum values
are not outliers and are important for future predictions. This
normalization technique can be applied with the following formula:

<div>

::: {#_idContainer037 .IMG---Figure}
![Figure 2.7: Formula for calculating normalization
](3_files/B15911_02_07.jpg)
:::

</div>

Figure 2.7: Formula for calculating normalization

Here, *O*[i]{.subscript} is the *i*[th]{.superscript} observation, *O*
represents a vector with all *O* values, and the functions *min (O)* and
*max (O)* represent the minimum and maximum values of the series,
respectively.

During *Exercise* *2.01*, *Exploring the Bitcoin Dataset and Preparing
Data for a Model*, we will prepare available Bitcoin data to be used in
our LSTM model. That includes selecting variables of interest, selecting
a relevant period, and applying the preceding point-relative
normalization technique.


Structuring Your Problem {#_idParaDest-55}
========================

::: {#_idContainer057 .Content}
Compared to researchers, practitioners spend much less time determining
which architecture to choose when starting a new deep learning project.
Acquiring data that represents a given problem correctly is the most
important factor to consider when developing these systems, followed by
an understanding of the dataset\'s inherent biases and limitations. When
starting to develop a deep learning system, consider the following
questions for reflection:

-   Do I have the right data? This is the hardest challenge when
    training a deep learning model. First, define your problem with
    mathematical rules. Use precise definitions and organize the problem
    into either categories (classification problems) or a continuous
    scale (regression problems). Now, how can you collect data
    pertaining to those metrics?
-   Do I have enough data? Typically, deep learning algorithms have
    shown to perform much better on large datasets than on smaller ones.
    Knowing how much data is necessary to train a high-performance
    algorithm depends on the kind of problem you are trying to address,
    but aim to collect as much data as you can.
-   Can I use a pre-trained model? If you are working on a problem that
    is a subset of a more general application, but within the same
    domain. Consider using a pre-trained model. Pre-trained models can
    give you a head start on tackling the specific patterns of your
    problem, instead of the more general characteristics of the domain
    at large. A good place to start is the official TensorFlow
    repository (<https://github.com/tensorflow/models>).

When you structure your problem with such questions, you will have a
sequential approach to any new deep learning project. The following is a
representative flow chart of these questions and tasks:

<div>

::: {#_idContainer038 .IMG---Figure}
![Figure 2.8: Decision tree of key reflection questions to be asked at
the beginning of a deep learning project ](4_files/B15911_02_08.jpg)
:::

</div>

Figure 2.8: Decision tree of key reflection questions to be asked at the
beginning of a deep learning project

In certain circumstances, the data may simply not be available.
Depending on the case, it may be possible to use a series of techniques
to effectively create more data from your input data. This process is
known as **data augmentation** and can be applied successfully when
working with image recognition problems.

Note

A good reference is the article *Classifying plankton with deep neural
networks*, available at
<https://benanne.github.io/2015/03/17/plankton.html>. The authors show a
series of techniques for augmenting a small set of image data in order
to increase the number of training samples the model has.

Once the problem is well-structured, you will be able to start preparing
the model.

[]{#_idTextAnchor056}

Jupyter Notebook {#_idParaDest-56}
----------------

We will be using Jupyter Notebook to code in this section. Jupyter
Notebooks provide Python sessions via a web browser that allows you to
work with data interactively. They are a popular tool for exploring
datasets. They will be used in exercises throughout this book.

[]{#_idTextAnchor057}

Exercise 2.01: Exploring the Bitcoin Dataset and Preparing Data for a Model {#_idParaDest-57}
---------------------------------------------------------------------------

In this exercise, we will prepare the data and then pass it to the
model. The prepared data will then be useful in making predictions as we
move ahead in the chapter. Before preparing the data, we will do some
visual analysis on it, such as looking at when the value of Bitcoin was
at its highest and when the decline started.

Note

We will be using a public dataset originally retrieved from the Yahoo
Finance website (<https://finance.yahoo.com/quote/BTC-USD/history/>).
The dataset has been slightly modified, as it has been provided
alongside this chapter, and will be used throughout the rest of this
book.

The dataset can be downloaded from: <https://packt.live/2Zgmm6r>.

The following are the steps to complete this exercise:

1.  Using your Terminal, navigate to the
    `Chapter02/Exercise2.01`{.literal} directory. Activate the
    environment created in the previous chapter and execute the
    following command to start a Jupyter Notebook instance:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ jupyter notebook
    ```
    :::

    This should automatically open the Jupyter lab server in your
    browser. From there you can start a Jupyter Notebook.

    You should see the following output or similar:

    ::: {#_idContainer039 .IMG---Figure}
    ![Figure 2.9: Terminal image after starting a Jupyter lab instance
    ](4_files/B15911_02_09.jpg)
    :::

    Figure 2.9: Terminal image after starting a Jupyter lab instance

2.  Select the `Exercise2.01_Exploring_Bitcoin_Dataset.ipynb`{.literal}
    file. This is a Jupyter Notebook file that will open in a new
    browser tab. The application will automatically start a new Python
    interactive session for you:

    ::: {#_idContainer040 .IMG---Figure}
    ![Figure 2.10: Landing page of your Jupyter Notebook instance
    ](4_files/B15911_02_10.jpg)
    :::

    Figure 2.10: Landing page of your Jupyter Notebook instance

3.  Click the Jupyter Notebook file:

    ::: {#_idContainer041 .IMG---Figure}
    ![Figure 2.11: Image of Jupyter Notebook ](4_files/B15911_02_11.jpg)
    :::

    Figure 2.11: Image of Jupyter Notebook

4.  Opening our Jupyter Notebook, consider the Bitcoin data made
    available with this chapter. The dataset
    `data/bitcoin_historical_prices.csv`{.literal}
    (<https://packt.live/2Zgmm6r>) contains the details of Bitcoin
    prices since early 2013. It contains eight variables, two of which
    (`date`{.literal} and `week`{.literal}) describe a time period of
    the data. These can be used as indices---and six others
    (`open`{.literal}, `high`{.literal}, `low`{.literal},
    `close`{.literal}, `volume`{.literal}, and
    `market_capitalization`{.literal}) can be used to understand changes
    in the price and value of Bitcoin over time:

    ::: {#_idContainer042 .IMG---Figure}
    ![Figure 2.12: Available variables (that is, columns) in the Bitcoin
    historical prices dataset ](4_files/B15911_02_12.jpg)
    :::

    Figure 2.12: Available variables (that is, columns) in the Bitcoin
    historical prices dataset

5.  Using the open Jupyter Notebook instance, consider the time series
    of two of those variables: `close`{.literal} and `volume`{.literal}.
    Start with those time series to explore price fluctuation patterns,
    that is, how the price of Bitcoin varied at different times in the
    historical data.

6.  Navigate to the open instance of the Jupyter Notebook,
    `Exercise2.01_Exploring_Bitcoin_Dataset.ipynb`{.literal}. Now,
    execute all cells under the `Introduction`{.literal} header. This
    will import the required libraries and import the dataset into
    memory:

    ::: {#_idContainer043 .IMG---Figure}
    ![Figure 2.13: Output from the first cells of the notebook
    time-series plot of the closing price for Bitcoin from the close
    variable](4_files/B15911_02_13.jpg)
    :::

    Figure 2.13: Output from the first cells of the notebook time-series
    plot of the closing price for Bitcoin from the close variable

7.  After the dataset has been imported into memory, move to the
    `Exploration`{.literal} section. You will find a snippet of code
    that generates a time series plot for the `close`{.literal}
    variable:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    bitcoin.set_index('date')['close'].plot(linewidth=2, \
                                            figsize=(14, 4),\
                                            color='#d35400')
    #plt.plot(bitcoin['date'], bitcoin['close'])
    ```
    :::

    The output looks like:

    ::: {#_idContainer044 .IMG---Figure}
    ![Figure 2.14: Time series plot of the closing price for Bitcoin
    from the close variable ](4_files/B15911_02_14.jpg)
    :::

    Figure 2.14: Time series plot of the closing price for Bitcoin from
    the close variable

8.  Reproduce this plot but using the `volume`{.literal} variable in a
    new cell below this one. You will have most certainly noticed that
    price variables surge in 2017 and then the downfall starts:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    bitcoin.set_index('date')['volume'].plot(linewidth=2, \
                                             figsize=(14, 4), \
                                             color='#d35400')
    ```
    :::

    ::: {#_idContainer045 .IMG---Figure}
    ![Figure 2.15: The total daily volume of Bitcoin coins
    ](4_files/B15911_02_15.jpg)
    :::

    Figure 2.15: The total daily volume of Bitcoin coins

    *Figure 2.15* shows that since 2017, Bitcoin transactions have
    significantly increased in the market. The total daily volume varies
    much more than daily closing prices.

9.  Execute the remaining cells in the Exploration section to explore
    the range from 2017 to 2018.

    Fluctuations in Bitcoin prices have been increasingly commonplace in
    recent years. While those periods could be used by a neural network
    to understand certain patterns, we will be excluding older
    observations, given that we are interested in predicting future
    prices for not-too-distant periods. Filter the data after 2016 only.
    Navigate to the `Preparing Dataset for a Model`{.literal} section.
    Use the pandas API to filter the data. Pandas provides an intuitive
    API for performing this operation.

10. Extract recent data and save it into a variable:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    bitcoin_recent = bitcoin[bitcoin['date'] >= '2016-01-04']
    ```
    :::

    The `bitcoin_recent`{.literal} variable now has a copy of our
    original Bitcoin dataset, but filtered to the observations that are
    newer or equal to January 4, 2016.

    Normalize the data using the point-relative normalization technique
    described in the *Data Normalization* section in the Jupyter
    Notebook. You will only nor[]{#_idTextAnchor058}malize two
    variables---`close`{.literal} and `volume`{.literal}---because those
    are the variables that we are working to predict.

11. Run the next cell in the notebook to ensure that we only keep the
    close and volume variables.

    In the same directory containing this chapter, we have placed a
    script called `normalizations.py`{.literal}. That script contains
    the three normalization techniques described in this chapter. We
    import that script into our Jupyter Notebook and apply the functions
    to our series.

12. Navigate to the `Preparing Dataset for a Model`{.literal} section.
    Now, use the `iso_week`{.literal} variable to group daily
    observations from a given week using the pandas
    `groupby()`{.literal} method. We can now apply the normalization
    function, `normalizations.point_relative_normalization()`{.literal},
    directly to the series within that week. We can store the
    normalization output as a new variable in the same pandas DataFrame
    using the following code:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    bitcoin_recent['close_point_relative_normalization'] = \
    bitcoin_recent.groupby('iso_week')['close']\
    .apply(lambda x: normalizations.point_relative_normalization(x))
    ```
    :::

13. The `close_point_relative_normalization`{.literal} variable now
    contains the normalized data for the `close`{.literal} variable:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    bitcoin_recent.set_index('date')\
    ['close_point_relative_normalization'].plot(linewidth=2, \
                                                figsize=(14,4), \
                                                color='#d35400')
    ```
    :::

    This will result in the following output:

    ::: {#_idContainer046 .IMG---Figure}
    ![Figure 2.16: Image of the Jupyter Notebook focusing on the section
    where the normalization function is applied
    ](4_files/B15911_02_16.jpg)
    :::

    Figure 2.16: Image of the Jupyter Notebook focusing on the section
    where the normalization function is applied

14. Do the same with the `volume`{.literal} variable
    (`volume_point_relative_normalization`{.literal}). The normalized
    `close`{.literal} variable contains an interesting variance pattern
    every week. We will be using that variable to train our LSTM model:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    bitcoin_recent.set_index('date')\
                            ['volume_point_relative_normalization'].\
                            plot(linewidth=2, \
                            figsize=(14,4), \
                            color='#d35400')
    ```
    :::

    Your output should be as follows.

    ::: {#_idContainer047 .IMG---Figure}
    ![Figure 2.17: Plot that displays the series from the normalized
    variable ](4_files/B15911_02_17.jpg)
    :::

    Figure 2.17: Plot that displays the series from the normalized
    variable

15. In order to evaluate how well the model performs, you need to test
    its accuracy versus some other data. Do this by creating two
    datasets: a training set and a test set. You will use 90 percent of
    the dataset to train the LSTM model and 10 percent to evaluate its
    performance. Given that the data is continuous and in the form of a
    time series, use the last 10 percent of available weeks as a test
    set and the first 90 percent as a training set:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    boundary = int(0.9 * bitcoin_recent['iso_week'].nunique())
    train_set_weeks = bitcoin_recent['iso_week'].unique()[0:boundary]
    test_set_weeks = bitcoin_recent[~bitcoin_recent['iso_week']\
                     .isin(train_set_weeks)]['iso_week'].unique()
    test_set_weeks
    train_set_weeks
    ```
    :::

    This will display the following output:

    ::: {#_idContainer048 .IMG---Figure}
    ![Figure 2.18: Output of the test set weeks
    ](4_files/B15911_02_18.jpg)
    :::

    Figure 2.18: Output of the test set weeks

    ::: {#_idContainer049 .IMG---Figure}
    ![Figure 2.19: Using weeks to create a training set
    ](4_files/B15911_02_19.jpg)
    :::

    Figure 2.19: Using weeks to create a training set

16. Create the separate datasets for each operation:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    train_dataset = bitcoin_recent[bitcoin_recent['iso_week']\
                                                 .isin(train_set_weeks)]
    test_dataset = bitcoin_recent[bitcoin_recent['iso_week'].\
                                                isin(test_set_weeks)]
    ```
    :::

17. Finally, navigate to the `Storing Output`{.literal} section and save
    the filtered variable to disk, as follows:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    test_dataset.to_csv('data/test_dataset.csv', index=False)
    train_dataset.to_csv('data/train_dataset.csv', index=False)
    bitcoin_recent.to_csv('data/bitcoin_recent.csv', index=False)
    ```
    :::

    Note

    To access the source code for this specific section, please refer to
    <https://packt.live/3ehbgCi>.

    You can also run this example online at
    <https://packt.live/2ZdGq9s>. You must execute the entire Notebook
    in order to get the desired result.

In this exercise, we explored the Bitcoin dataset and prepared it for a
deep learning model.

We learned that in 2017, the price of Bitcoin skyrocketed. This
phenomenon took a long time to take place and may have been influenced
by a number of external factors that this data alone doesn\'t explain
(for instance, the emergence of other cryptocurrencies). After the great
surge of 2017, we saw a great fall in the value of Bitcoin in 2018.

We also used the point-relative normalization technique to process the
Bitcoin dataset in weekly chunks. We do this to train an LSTM network to
learn the weekly patterns of Bitcoin price changes so that it can
predict a full week into the future.

However, Bitcoin statistics show significant fluctuations on a weekly
basis. Can we predict the price of Bitcoin in the future? What will the
price be seven days from now? We will build a deep learning model to
explore these questions in our next section using Keras.


Using Keras as a TensorFlow Interface {#_idParaDest-58}
=====================================

::: {#_idContainer057 .Content}
We are using Keras because it simplifies the TensorFlow interface into
general abstractions and, in TensorFlow 2.0, this is the default API in
this version. In the backend, the computations are still performed in
TensorFlow, but we spend less time worrying about individual components,
such as variables and operations, and spend more time building the
network as a computational unit. Keras makes it easy to experiment with
different architectures and hyperparameters, moving more quickly toward
a performant solution.

As of TensorFlow 2.0.0, Keras is now officially distributed with
TensorFlow as `tf.keras`{.literal}. This suggests that Keras is now
tightly integrated with TensorFlow and will likely continue to be
developed as an open source tool for a long period of time. Components
are an integral part when building models. Let\'s deep dive into this
concept now.

[]{#_idTextAnchor060}

Model Components {#_idParaDest-59}
----------------

As we saw in *Chapter 1*, *Introduction to Neural Networks and Deep
Learning*, LSTM networks also have input, hidden, and output layers.
Each hidden layer has an activation function that evaluates that
layer\'s associated weights and biases. As expected, the network moves
data sequentially from one layer to another and evaluates the results by
the output at every iteration (that is, an epoch).

Keras provides intuitive classes that represent each one of the
components listed in the following table:

<div>

::: {#_idContainer050 .IMG---Figure}
![Figure 2.20: Description of key components from the Keras API
](5_files/B15911_02_20.jpg)
:::

</div>

Figure 2.20: Description of key components from the Keras API

We will be using these components to build a deep learning model.

Keras\' `keras.models.Sequential()`{.literal} component represents a
whole sequential neural network. This Python class can be instantiated
on its own and have other components added to it subsequently.

We are interested in building an LSTM network because those networks
perform well with sequential data---and a time series is a kind of
sequential data. Using Keras, the complete LSTM network would be
implemented as follows:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation
model = Sequential()
model.add(LSTM(units=number_of_periods, \
               input_shape=(period_length, number_of_periods) \
               return_sequences=False), stateful=True)
model.add(Dense(units=period_length)) \
          model.add(Activation("linear"))
model.compile(loss="mse", optimizer="rmsprop")
```
:::

This implementation will be further optimized in *Chapter 3*,
*Real-World Deep Learning with TensorFlow and Keras: Evaluating the
Bitcoin Model*.

Keras abstraction allows you to focus on the key elements that make a
deep learning system more performant: determining the right sequence of
components, how many layers and nodes to include, and which activation
function to use. All of these choices are determined by either the order
in which components are added to the instantiated
`keras.models`{.literal}`.Sequential()`{.literal} class or by parameters
passed to each component instantiation (that is,
`Activation("linear")`{.literal}). The final `model.compile()`{.literal}
step builds the neural network using TensorFlow components.

After the network is built, we train our network using the
`model.fit()`{.literal} method. This will yield a trained model that can
be used to make predictions:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model.fit(X_train, Y_train,
          batch_size=32, epochs=epochs)
```
:::

The `X_train`{.literal} and `Y_train`{.literal} variables are,
respectively, a set used for training and a smaller set used for
evaluating the loss function (that is, testing how well the network
predicts data). Finally, we can make predictions using the
`model.predict()`{.literal} method:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model.predict(x=X_train)
```
:::

The preceding steps cover the Keras paradigm for working with neural
networks. Despite the fact that different architectures can be dealt
with in very different ways, Keras simplifies the interface for working
with different architectures by using three components --
`Network Architecture`{.literal}, `Fit`{.literal}, and
`Predict`{.literal}:

<div>

::: {#_idContainer051 .IMG---Figure}
![Figure 2.21: The Keras neural network paradigm
](5_files/B15911_02_21.jpg)
:::

</div>

Figure 2.21: The Keras neural network paradigm

The Keras neural network diagram comprises the following three steps:

-   A neural network architecture
-   Training a neural network (or **Fit**)
-   Making predictions

Keras allows much greater control within each of these steps. However,
its focus is to make it as easy as possible for users to create neural
networks in as little time as possible. That means that we can start
with a simple model, and then add complexity to each one of the
preceding steps to make that initial model perform better.

We will take advantage of that paradigm during our upcoming exercise and
chapters. In the next exercise, we will create the simplest LSTM network
possible. Then, in *Chapter 3*, *Real-World Deep Learning: Evaluating
the Bitcoin Model*, we will continuously evaluate and alter that network
to make it more robust and performant.

[]{#_idTextAnchor061}

Exercise 2.02: Creating a TensorFlow Model Using Keras {#_idParaDest-60}
------------------------------------------------------

In this notebook, we design and compile a deep learning model using
Keras as an interface to TensorFlow. We will continue to modify this
model in our next chapters and exercises by experimenting with different
optimization techniques. However, the essential components of the model
are designed entirely in this notebook:

1.  Open a new Jupyter Notebook and import the following libraries:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense, Activation
    ```
    :::

2.  Our dataset contains daily observations and each observation
    influences a future observation. Also, we are interested in
    predicting a week---that is, 7 days---of Bitcoin prices in the
    future:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    period_length = 7
    number_of_periods = 208 - 21 - 1
    number_of_periods
    ```
    :::

    We have calculated `number_of_observations`{.literal} based on
    available weeks in our dataset. Given that we will be using last
    week to test the LSTM network on every epoch, we will use 208 -- 21
    -- 1. You\'ll get:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    186
    ```
    :::

3.  Build the LSTM model using Keras. We have the batch size as one
    because we are passing the whole data in a single iteration. If data
    is big, then we can pass the data with multiple batches, That\'s why
    we used batch\_input\_shape:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def build_model(period_length, number_of_periods, batch_size=1):
        model = Sequential()
        model.add(LSTM(units=period_length,\
                       batch_input_shape=(batch_size, \
                                          number_of_periods, \
                                          period_length),\
                       input_shape=(number_of_periods, \
                                    period_length),\
                       return_sequences=False, stateful=False))
        model.add(Dense(units=period_length))
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer="rmsprop")
        return model
    ```
    :::

    This should return a compiled Keras model that can be trained and
    stored in disk.

4.  Let\'s store the model on the output of the model to a disk:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    model = build_model(period_length=period_length, \
                        number_of_periods=number_of_periods)
    model.save('bitcoin_lstm_v0.h5')
    ```
    :::

    Note that the `bitcoin_lstm_v0.h5`{.literal} model hasn\'t been
    trained yet. When saving a model without prior training, you
    effectively only save the architecture of the model. That same model
    can later be loaded by using Keras\' `load_model()`{.literal}
    function, as follows:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    model = keras.models.load_model('bitcoin_lstm_v0.h5')
    ```
    :::

    Note

    To access the source code for this specific section, please refer to
    <https://packt.live/38KQI3Y>.

    You can also run this example online at
    <https://packt.live/3fhEL89>. You must execute the entire Notebook
    in order to get the desired result.

This concludes the creation of our Keras model, which we can now use to
make predictions.

Note

You may encounter the following warning when loading the Keras library:

`Using TensorFlow backend`{.literal}

Keras can be configured to use another backend instead of TensorFlow
(that is, Theano). In order to avoid this message, you can create a file
called `keras.json`{.literal} and configure its backend there. The
correct configuration of that file depends on your system. Hence, it is
recommended that you visit Keras\' official documentation on the topic
at <https://keras.io/backend/>.

In this section, we have learned how to build a deep learning model
using Keras---an interface for TensorFlow. We studied core components of
Keras and used those components to build the first version of our
Bitcoin price-predicting system based on an LSTM model.

In our next section, we will discuss how to put all the components from
this chapter together into a (nearly complete) deep learning system.
That system will yield our very first predictions, serving as a starting
point for future improvements.


From Data Preparation to Modeling {#_idParaDest-61}
=================================

::: {#_idContainer057 .Content}
This section focuses on the implementation aspects of a deep learning
system. We will use the Bitcoin data from the *Choosing the Right Model
Architecture* section, and the Keras knowledge from the preceding
section, *Using Keras as a TensorFlow Interface*, to put both of these
components together. This section concludes the chapter by building a
system that reads data from a disk and feeds it into a model as a single
piece of software.

[]{#_idTextAnchor063}

Training a Neural Network {#_idParaDest-62}
-------------------------

Neural networks can take long periods of time to train. Many factors
affect how long that process may take. Among them, three factors are
commonly considered the most important:

-   The network\'s architecture
-   How many layers and neurons the network has
-   How much data there is to be used in the training process

Other factors may also greatly impact how long a network takes to train,
but most of the optimization that a neural network can have when
addressing a business problem comes from exploring those three.

We will be using the normalized data from our previous section. Recall
that we have stored the training data in a file called
`train_dataset.csv`{.literal}.

Note:

You can download the training data by visiting this link:
<https://packt.live/2Zgmm6r>.

We will load that dataset into memory using the `pandas`{.literal}
library for easy exploration:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
import pandas as pd
train = pd.read_csv('data/train_dataset.csv')
```
:::

Note

Make sure you change the path (highlighted) based on where you have
downloaded or saved the CSV file.

You will see the output in a tabular form as follows:

<div>

::: {#_idContainer052 .IMG---Figure}
![Figure 2.22: Table showing the first five rows of the training dataset
](6_files/B15911_02_22.jpg)
:::

</div>

Figure 2.22: Table showing the first five rows of the training dataset

We will be using the series from the
`close_point_relative_normalization`{.literal} variable, which is a
normalized series of the Bitcoin closing prices---from the
`close`{.literal} variable---since the beginning of 2016.

The `close_point_relative_normalization`{.literal} variable has been
normalized on a weekly basis. Each observation from the week\'s period
is made relative to the difference from the closing prices on the first
day of the period. This normalization step is important and will help
our network train faster:

<div>

::: {#_idContainer053 .IMG---Figure}
![Figure 2.23: Plot that displays the series from the normalized
variable. ](6_files/B15911_02_23.jpg)
:::

</div>

Figure 2.23: Plot that displays the series from the normalized variable.

This variable will be used to train our LSTM model.

[]{#_idTextAnchor064}

Reshaping Time Series Data {#_idParaDest-63}
--------------------------

Neural networks typically work with vectors and tensors---both
mathematical objects that organize data in a number of dimensions. Each
neural network implemented in Keras will have either a vector or a
tensor that is organized according to a specification as input.

At first, understanding how to reshape the data into the format expected
by a given layer can be confusing. To avoid confusion, it is advisable
to start with a network with as few components as possible, and then add
components gradually. Keras\' official documentation (under the
`Layers`{.literal} section) is essential for learning about the
requirements for each kind of layer.

Note

The Keras official documentation is available at
<https://keras.io/layers/core/>. That link takes you directly to the
`Layers`{.literal} section.

**NumPy** is a popular Python library used for performing numerical
computations. It is used by the deep learning community to manipulate
vectors and tensors and prepare them for deep learning systems.

In particular, the `numpy.reshape()`{.literal} method is very important
when adapting data for deep learning models. That model allows for the
manipulation of NumPy arrays, which are Python objects analogous to
vectors and tensors.

We\'ll now organize the prices from the
`close_point_relative_normalization`{.literal} variable using the weeks
after 2016. We will create distinct groups containing 7 observations
each (one for each day of the week) for a total of 208 complete weeks.
We do that because we are interested in predicting the prices of a
week\'s worth of trading.

Note

We use the ISO standard to determine the beginning and the end of a
week. Other kinds of organizations are entirely possible. This one is
simple and intuitive to follow, but there is room for improvement.

LSTM networks work with three-dimensional tensors. Each one of those
dimensions represents an important property for the network. These
dimensions are as follows:

-   **Period length**: The period length, that is, how many observations
    there are for a period
-   **Number of periods**: How many periods are available in the dataset
-   **Number of features**: The number of features available in the
    dataset

Our data from the `close_point_relative_normalization`{.literal}
variable is currently a one-dimensional vector. We need to reshape it to
match those three dimensions.

We will be using a period of a week. So, our period length is 7 days
(period length = 7). We have 208 complete weeks available in our data.
We will be using the very last of those weeks to test our model against
during its training period. That leaves us with 187 distinct weeks.
Finally, we will be using a single feature in this network (number of
features = 1); we will include more features in future versions.

To reshape the data to match those dimensions, we will be using a
combination of base Python properties and the `reshape()`{.literal}
method from the `numpy`{.literal} library. First, we create the 186
distinct week groups with 7 days, each using pure Python:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
group_size = 7
samples = list()
for i in range(0, len(data), group_size):
sample = list(data[i:i + group_size])
         if len(sample) == group_size:samples\
                           .append(np.array(sample)\
                           .reshape(group_size, 1).tolist())
data = np.array(samples)
```
:::

This piece of code creates distinct week groups. The resulting variable
data is a variable that contains all the right dimensions.

Note

Each Keras layer will expect its input to be organized in specific ways.
However, Keras will reshape data accordingly, in most cases. Always
refer to the Keras documentation on layers
(<https://keras.io/layers/core/>) before adding a new layer.

The Keras LSTM layer expects these dimensions to be organized in a
specific order: the number of features, the number of observations, and
the period length. Reshape the dataset to match that format:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
X_train = data[:-1,:].reshape(1, 186, 7)
Y_validation = data[-1].reshape(1, 7)
```
:::

The preceding snippet also selects the very last week of our set as a
validation set (via `data[-1]`{.literal}). We will be attempting to
predict the very last week in our dataset by using the preceding 76
weeks. The next step is to use those variables to fit our model:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model.fit(x=X_train, y=Y_validation, epochs=100)
```
:::

LSTMs are computationally expensive models. They may take up to 5
minutes to train with our dataset on a modern computer. Most of that
time is spent at the beginning of the computation when the algorithm
creates the full computation graph. The process gains speed after it
starts training:

<div>

::: {#_idContainer054 .IMG---Figure}
![Figure 2.24: Graph that shows the results of the loss function
evaluated at each epoch ](6_files/B15911_02_24.jpg)
:::

</div>

Figure 2.24: Graph that shows the results of the loss function evaluated
at each epoch

Note

This compares what the model predicted at each epoch, and then compares
that with the real data using a technique called mean-squared error.
This plot shows those results.

At a glance, our network seems to perform very well; it starts with a
very small error rate that continuously decreases. Now that we have
lowered the error rate, let\'s move on to make some predictions.

[]{#_idTextAnchor065}

Making Predictions {#_idParaDest-64}
------------------

After our network has been trained, we can proceed to make predictions.
We will be making predictions for a future week beyond our time period.

Once we have trained our model with the `model.fit()`{.literal} method,
making predictions is simple:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model.predict(x=X_train)
```
:::

We use the same data for making predictions as the data used for
training (the `X_train`{.literal} variable). If we have more data
available, we can use that instead---given that we reshape it to the
format the LSTM requires.

[]{#_idTextAnchor066}

### Overfitting {#_idParaDest-65}

When a neural network overfits a validation set, this means that it
learns patterns present in the training set but is unable to generalize
it to unseen data (for instance, the test set). In the next chapter, we
will learn how to avoid overfitting and create a system for both
evaluating our network and increasing its performance:

<div>

::: {#_idContainer055 .IMG---Figure}
![Figure 2.25: Graph showing the weekly performance of Bitcoin
](6_files/B15911_02_25.jpg)
:::

</div>

Figure 2.25: Graph showing the weekly performance of Bitcoin

In the plot shown above, the horizontal axis represents the week number
and the vertical axis represents the predicted performance of Bitcoin.
Now that we have explored the data, prepared a model, and learned how to
make predictions, let\'s put this knowledge into practice.

[]{#_idTextAnchor067}

Activity 2.01: Assembling a Deep Learning System {#_idParaDest-66}
------------------------------------------------

In this activity, we\'ll bring together all the essential pieces for
building a basic deep learning system -- the data, a model, and
predictions:

1.  Start a Jupyter Notebook.
2.  Load the training dataset into memory.
3.  Inspect the training set to see whether it is in the form period
    length, number of periods, or number of features.
4.  Convert the training set if it is not in the required format.
5.  Load the previously trained model.
6.  Train the model using your training dataset.
7.  Make a prediction on the training set.
8.  Denormalize the values and save the model.

The final output will look as follows with the horizontal axis
representing the number of days and the vertical axis represents the
price of Bitcoin:

6

<div>

::: {#_idContainer056 .IMG---Figure}
![Figure 2.26: Expected output ](6_files/B15911_02_26.jpg)
:::

</div>

Figure 2.26: Expected output

Note

The solution to this activity can be found on page 136.


Summary {#_idParaDest-67}
=======

::: {#_idContainer057 .Content}
In this chapter, we have assembled a complete deep learning system, from
data to prediction. The model created in this activity requires a number
of improvements before it can be considered useful. However, it serves
as a great starting point from which we will continuously improve.

The next chapter will explore techniques for measuring the performance
of our model and will continue to make modifications until we reach a
model that is both useful and robust.
