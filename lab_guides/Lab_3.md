
3. Real-World Deep Learning: Evaluating the Bitcoin Model {#_idParaDest-68}
=========================================================

::: {#_idContainer060 .Content}
:::

::: {#_idContainer081 .Content}
[]{#_idTextAnchor070}Overview

This chapter focuses on how to evaluate a neural network model. We\'ll
modify the network\'s hyperparameters to improve its performance.
However, before altering any parameters, we need to measure how the
model performs. By the end of this chapter, you will be able to evaluate
a model using different functions and techniques. You will also learn
about hypermeter optimization by implementing functions and
regularization strategies.


Introduction {#_idParaDest-69}
============

::: {#_idContainer081 .Content}
In the previous chapter, you trained your model. But how will you check
its performance and whether it is performing well or not? Let\'s find
out by evaluating a model. In machine learning, it is common to define
two distinct terms: **parameter** and **hyperparameter**. Parameters are
properties that affect how a model makes predictions from data, say from
a particular dataset. Hyperparameters refer to how a model learns from
data. Parameters can be learned from the data and modified dynamically.
Hyperparameters, on the other hand, are higher-level properties defined
before the training begins and are not typically learned from data. In
this chapter, you will learn about these factors in detail and
understand how to use them with different evaluation techniques to
improve the performance of a model.

Note

For a more detailed overview of machine learning, refer to *Python
Machine Learning*, *Sebastian Raschka and Vahid Mirjalili, Packt
Publishing, 2017)*.

[]{#_idTextAnchor072}

Problem Categories {#_idParaDest-70}
------------------

Generally, there are two categories of problems that can be solved by
neural networks: **classification** and **regression**. Classification
problems concern the prediction of the right categories from data---for
instance, whether the temperature is hot or cold. Regression problems
are about the prediction of values in a continuous scalar---for
instance, what the actual temperature value is.

The problems in these two categories are characterized by the following
properties:

-   **Classification**: These are problems that are characterized by
    categories. The categories can be different, or not. They can also
    be about a binary problem, where the outcome can either be
    `yes`{.literal} or `no`{.literal}. However, they must be clearly
    assigned to each data element. An example of a classification
    problem would be to assign either the `car`{.literal} or
    `not a car`{.literal} labels to an image using a convolutional
    neural network. The MNIST example we explored in *Chapter 1*,
    *Introduction to Neural Networks and Deep Learning*, is another
    example of a classification problem.
-   **Regression**: These are problems that are characterized by a
    continuous variable (that is, a scalar). They are measured in terms
    of ranges, and their evaluations consider how close to the real
    values the network is. An example is a time-series classification
    problem in which a recurrent neural network is used to predict the
    future temperature values. The Bitcoin price-prediction problem is
    another example of a regression problem.

While the overall structure of how to evaluate these models is the same
for both of these problem categories, we employ different techniques to
evaluate how models perform. In the next section, we\'ll explore these
techniques for either classification or regression problems.

[]{#_idTextAnchor073}

Loss Functions, Accuracy, and Error Rates {#_idParaDest-71}
-----------------------------------------

Neural networks utilize functions that measure how the networks perform
compared to a **validation set**---that is, a part of the data that is
kept separate to be used as part of the training process. These
functions are called **loss functions**.

Loss functions evaluate how wrong a neural network\'s predictions are.
Then, they propagate those errors back and make adjustments to the
network, modifying how individual neurons are activated. Loss functions
are key components of neural networks, and choosing the right loss
function can have a significant impact on how the network performs.
Errors are propagated through a process called **backpropagation**,
which is a technique for propagating the errors that are returned by the
loss function to each neuron in a neural network. Propagated errors
affect how neurons activate, and ultimately, how they influence the
output of that network.

Many neural network packages, including Keras, use this technique by
default.

Note

For more information about the mathematics of backpropagation, please
refer to *Deep Learning* by *Ian Goodfellow et. al., MIT Press, 2016*.

We use different loss functions for regression and classification
problems. For classification problems, we use accuracy functions (that
is, the proportion of the number of times the predictions were correct
to the number of times predictions were made). For example, if you
predict a toss of a coin that will result in *m* times as heads when you
toss it *n* times and your prediction is correct, then the accuracy will
be calculated as *m/n*. For regression problems, we use error rates
(that is, how close the predicted values were to the observed ones).

Here\'s a summary of common loss functions that can be utilized,
alongside their common applications:

<div>

::: {#_idContainer061 .IMG---Figure}
![Figure 3.1: Common loss functions used for classification and
regression problems ](2_files/B15911_03_01.jpg)
:::

</div>

Figure 3.1: Common loss functions used for classification and regression
problems

For regression problems, the MSE function is the most common choice,
while for classification problems, binary cross-entropy (for binary
category problems) and categorical cross-entropy (for multi-category
problems) are common choices. It is advised to start with these loss
functions, then experiment with other functions as you evolve your
neural network, aiming to gain performance.

The network we developed in *Chapter 2*, *Real-World Deep Learning with
TensorFlow and Keras: Predicting the Price of Bitcoin*, uses MSE as its
loss function. In the next section, we\'ll explore how that function
performs as the network trains.

[]{#_idTextAnchor074}

Different Loss Functions, Same Architecture {#_idParaDest-72}
-------------------------------------------

Before moving ahead to the next section, let\'s explore, in practical
terms, how these problems are different in the context of neural
networks.

The *TensorFlow Playground* (<https://playground.tensorflow.org/>)
application has been made available by the TensorFlow team to help us
understand how neural networks work. Here, we can see a neural network
represented with its layers: input (on the left), hidden layers (in the
middle), and output (on the right).

Note:

These images can be viewed in the repository on GitHub at:
<https://packt.live/2Cl1t0H>.

We can also choose different sample datasets to experiment with on the
far left-hand side. And, finally, on the far right-hand side, we can see
the output of the network:

<div>

::: {#_idContainer062 .IMG---Figure}
![Figure 3.2: TensorFlow Playground web application
](2_files/B15911_03_02.jpg)
:::

</div>

Figure 3.2: TensorFlow Playground web application

Take the parameters for a neural network shown in this visualization to
gain an idea of how each parameter affects the model\'s results. This
application helps us explore the different problem categories we
discussed in the previous section. When we choose `Regression`{.literal}
(upper right-hand corner), the colors of the dots are colored in a range
of color values between orange and blue:

<div>

::: {#_idContainer063 .IMG---Figure}
![Figure 3.3: Regression problem example in TensorFlow Playground
](2_files/B15911_03_03.jpg)
:::

</div>

Figure 3.3: Regression problem example in TensorFlow Playground

When we choose `Classification`{.literal} as the
`Problem type`{.literal}, the dots in the dataset are colored with only
two color values: either blue or orange. When working on classification
problems, the network evaluates its loss function based on how many
blues and oranges the network has gotten wrong. It checks how far away
to the right the color values are for each dot in the network, as shown
in the following screenshot:

<div>

::: {#_idContainer064 .IMG---Figure}
![Figure 3.4: Details of the TensorFlow Playground application.
Different colors are assigned to different classes in classification
problems ](2_files/B15911_03_04.jpg)
:::

</div>

Figure 3.4: Details of the TensorFlow Playground application. Different
colors are assigned to different classes in classification problems

After clicking on the play button, we notice that the numbers in the
`Training loss`{.literal} area keep going down as the network
continuously trains. The numbers are very similar in each problem
category because the loss functions play the same role in both neural
networks.

However, the actual loss function that\'s used for each category is
different and is chosen depending on the problem type.

[]{#_idTextAnchor075}

Using TensorBoard {#_idParaDest-73}
-----------------

Evaluating neural networks is where TensorBoard excels. As we explained
in *Chapter 1*, *Introduction to Neural Networks and Deep Learning*,
TensorBoard is a suite of visualization tools that\'s shipped with
TensorFlow. Among other things, we can explore the results of loss
function evaluations after each epoch. A great feature of TensorBoard is
that we can organize the results of each run separately and compare the
resulting loss function metrics for each run. We can then decide on
which hyperparameters to tune and have a general sense of how the
network is performing. The best part is that this is all done in real
time.

In order to use TensorBoard with our model, we will use Keras\'
`callback`{.literal} function. We do this by importing the TensorBoard
`callback`{.literal} and passing it to our model when calling its
`fit()`{.literal} function. The following code shows an example of how
this would be implemented in the Bitcoin model we created in the
*Chapter 2*, *Real-World Deep Learning with TensorFlow and Keras:
Predicting the Price of Bitcoin*:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
from tensorflow.keras.callbacks import TensorBoard
model_name = 'bitcoin_lstm_v0_run_0'
tensorboard = TensorBoard(log_dir='logs\\{}'.format(model_name)) \
                          model.fit(x=X_train, y=Y_validate, \
                          batch_size=1, epochs=100, verbose=0, \
                          callbacks=[tensorboard])
```
:::

Keras `callback`{.literal} functions are called at the end of each epoch
run. In this case, Keras calls the TensorBoard `callback`{.literal} to
store the results from each run on the disk. There are many other useful
`callback`{.literal} functions available, and you can create custom ones
using the Keras API.

Note

Please refer to the Keras callback documentation
(<https://keras.io/callbacks/>) for more information.

After implementing the TensorBoard callback, the loss function metrics
are now available in the TensorBoard interface. You can now run a
TensorBoard process (with `tensorboard --logdir=./logs`{.literal}) and
leave it running while you train your network with `fit()`{.literal}.

The main graphic to evaluate is typically called loss. We can add more
metrics by passing known metrics to the metrics parameter in the
`fit()`{.literal} function. These will then be available for
visualization in TensorBoard, but will not be used to adjust the network
weights. The interactive graphics will continue to update in real time,
which allows you to understand what is happening on every epoch. In the
following screenshot, you can see a TensorBoard instance showing loss
function results, alongside other metrics that have been added to the
metrics parameter:

<div>

::: {#_idContainer065 .IMG---Figure}
![Figure 3.5: Screenshot of a TensorBoard instance showing the loss
function results ](2_files/B15911_03_05.jpg)
:::

</div>

Figure 3.5: Screenshot of a TensorBoard instance showing the loss
function results

In the next section, we will talk more about how to implement the
different metrics we discussed in this section.

[]{#_idTextAnchor076}

Implementing Model Evaluation Metrics {#_idParaDest-74}
-------------------------------------

In both regression and classification problems, we split the input
dataset into three other datasets: train, validation, and test. Both the
train and the validation sets are used to train the network. The train
set is used by the network as input, while the validation set is used by
the loss function to compare the output of the neural network to the
real data and compute how wrong the predictions are. Finally, the test
set is used after the network has been trained to measure how the
network can perform on data it has never seen before.

Note

There isn\'t a clear rule for determining how the train, validation, and
test datasets must be divided. It is a common approach to divide the
original dataset into 80 percent train and 20 percent test, then to
further divide the train dataset into 80 percent train and 20 percent
validation. For more information about this problem, please refer to
*Python Machine Learning*, by *Sebastian Raschka and Vahid Mirjalili
(Packt Publishing, 2017)*.

In classification problems, you pass both the data and the labels to the
neural network as related but distinct data. The network then learns how
the data is related to each label. In regression problems, instead of
passing data and labels, we pass the variable of interest as one
parameter and the variables that are used for learning patterns as
another. Keras provides an interface for both of those use cases with
the `fit()`{.literal} method.

Note

The `fit()`{.literal} method can use either the
`validation_split`{.literal} or the `validation_data`{.literal}
parameter, but not both at the same time.

See the following snippet to understand how to use the
`validation_split`{.literal} and `validation_data`{.literal} parameters:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model.fit(x=X_train, y=Y_ train, \
          batch_size=1, epochs=100, verbose=0, \
          callbacks=[tensorboard], validation_split=0.1, \
          validation_data=(X_validation, Y_validation))
```
:::

`X_train`{.literal}: features from training set

`Y_train`{.literal}: labels from training set

`batch_size`{.literal}: the size of one batch

`epochs`{.literal}: the number of iterations

`verbose`{.literal}: the level of output you want

`callbacks`{.literal}: call a function after every epoch

`validation_split`{.literal}: validation percentage split if you have
not created it explicitly

`validation_data`{.literal}: validation data if you have created it
explicitly

Loss functions evaluate the progress of models and adjust their weights
on every run. However, loss functions only describe the relationship
between training data and validation data. In order to evaluate if a
model is performing correctly, we typically use a third set of
data---which is not used to train the network---and compare the
predictions made by our model to the values available in that set of
data. That is the role of the test set.

Keras provides the `model.evaluate()`{.literal} method, which makes the
process of evaluating a trained neural network against a test set easy.
The following code illustrates how to use the `evaluate()`{.literal}
method:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model.evaluate(x=X_test, y=Y_test)
```
:::

The `evaluate()`{.literal} method returns both the results of the loss
function and the results of the functions passed to the
`metrics`{.literal} parameter. We will be using that function frequently
in the Bitcoin problem to test how the model performs on the test set.

You will notice that the Bitcoin model we trained previously looks a bit
different than this example. That is because we are using an LSTM
architecture. LSTMs are designed to predict sequences.

Because of that, we do not use a set of variables to predict a different
single variable---even if it is a regression problem. Instead, we use
previous observations from a single variable (or set of variables) to
predict future observations of that same variable (or set). The
`y`{.literal} parameter on `keras.fit()`{.literal} contains the same
variable as the `x`{.literal} parameter, but only the predicted
sequences. So, let\'s have a look at how to evaluate the bitcoin model
we trained previously.

[]{#_idTextAnchor077}

Evaluating the Bitcoin Model {#_idParaDest-75}
----------------------------

We created a test set during our activities in *Chapter 1*,
*Introduction to Neural Networks and Deep Learning*. That test set
contains 21 weeks of daily Bitcoin price observations, which is
equivalent to about 10 percent of the original dataset.

We also trained our neural network using the other 90 percent of data
(that is, the train set with 187 weeks of data, minus 1 for the
validation set) in *Chapter 2*, *Real-World Deep Learning with
TensorFlow and Keras: Predicting the Price of Bitcoin*, and stored the
trained network on disk (`bitcoin_lstm_v0`{.literal}). We can now use
the `evaluate()`{.literal} method in each of the 21 weeks of data from
the test set and inspect how that first neural network performs.

In order to do that, though, we have to provide 186 preceding weeks. We
have to do this because our network has been trained to predict one week
of data using exactly 186 weeks of continuous data (we will deal with
this behavior by retraining our network periodically with larger periods
in *Chapter 4*, *Productization*, when we deploy a neural network as a
web application). The following snippet implements the
`evaluate()`{.literal} method to evaluate the performance of our model
in a test dataset:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
combined_set = np.concatenate((train_data, test_data), axis=1) \
               evaluated_weeks = []
for i in range(0, validation_data.shape[1]):
    input_series = combined_set[0:,i:i+187]
X_test = input_series[0:,:-1].reshape(1, \
         input_series.shape[1] – 1, ) \
         Y_test = input_series[0:,-1:][0]
result = B.model.evaluate(x=X_test, y=Y_test, verbose=0) \
         evaluated_weeks.append(result)
```
:::

In the preceding code, we evaluate each week using Keras\'
`model.evaluate()`{.literal} method, then store its output in the
`evaluated_weeks`{.literal} variable. We then plot the resulting MSE for
each week, as shown in the following plot:

<div>

::: {#_idContainer066 .IMG---Figure}
![Figure 3.6: MSE for each week in the test set
](2_files/B15911_03_06.jpg)
:::

</div>

Figure 3.6: MSE for each week in the test set

The resulting MSE from our model suggests that our model performs well
during most weeks, except for weeks 2, 8, 12, and 16, when its value
increases from about 0.005 to 0.02. Our model seems to be performing
well for almost all of the other test weeks.

[]{#_idTextAnchor078}

Overfitting {#_idParaDest-76}
-----------

Our first trained network (`bitcoin_lstm_v0`{.literal}) may be suffering
from a phenomenon known as **overfitting**. Overfitting is when a model
is trained to optimize a validation set, but it does so at the expense
of more generalizable patterns from the phenomenon we are interested in
predicting. The main issue with overfitting is that a model learns how
to predict the validation set, but fails to predict new data.

The loss function we used in our model reaches very low levels at the
end of our training process. Not only that, but this happens early: the
MSE loss function that\'s used to predict the last week in our data
decreases to a stable plateau around epoch 30. This means that our model
is predicting the data from week 187 almost perfectly, using the
preceding 186 weeks. Could this be the result of overfitting?

Let\'s look at the preceding plot again. We know that our LSTM model
reaches extremely low values in our validation set (about 2.9 ×
10[-6]{.superscript}), yet it also reaches low values in our test set.
The key difference, however, is in the scale. The MSE for each week in
our test set is about 4,000 times bigger (on average) than in the test
set. This means that the model is performing much worse in our test data
than in the validation set. This is worth considering.

The scale, though, hides the power of our LSTM model; even performing
much worse in our test set, the predictions\' MSE errors are still very,
very low. This suggests that our model may be learning patterns from the
data.

[]{#_idTextAnchor079}

Model Predictions {#_idParaDest-77}
-----------------

It\'s one thing is to measure our model comparing MSE errors, and
another to be able to interpret its results intuitively.

Using the same model, let\'s create a series of predictions for the
following weeks, using 186 weeks as input. We do that by sliding a
window of 186 weeks over the complete series (that is, train plus test
sets) and making predictions for each of those windows.

The following snippet makes predictions for all the weeks of the test
dataset using the `Keras model.predict()`{.literal} method:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
combined_set = np.concatenate((train_data, test_data), \
               axis=1) predicted_weeks = []
for i in range(0, validation_data.shape[1] + 1): 
    input_series = combined_set[0:,i:i+186]
    predicted_weeks.append(B.predict(input_series))
```
:::

In the preceding code, we make predictions using the
`model.predict()`{.literal} method, then store these predictions in the
`predicted_weeks`{.literal} variable. Then, we plot the resulting
predictions, making the following plot:

<div>

::: {#_idContainer067 .IMG---Figure}
![Figure 3.7: MSE for each week in the test set
](2_files/B15911_03_07.jpg)
:::

</div>

Figure 3.7: MSE for each week in the test set

The results of our model suggest that its performance isn\'t all that
bad. By observing the pattern from the `Predicted`{.literal} line
(grey), we can see that the network has identified a fluctuating pattern
happening on a weekly basis, in which the normalized prices go up in the
middle of the week, then down by the end of it but. However, there\'s
still a lot of room for improvement as it is unable to pick up higher
fluctuations. With the exception of a few weeks---the same as with our
previous MSE analysis---most weeks fall close to the correct values.

Now, let\'s denormalize the predictions so that we can investigate the
prediction values using the same scale as the original data (that is, US
dollars). We can do this by implementing a denormalization function that
uses the day index from the predicted data to identify the equivalent
week on the test data. After that week has been identified, the function
then takes the first value of that week and uses that value to
denormalize the predicted values by using the same point-relative
normalization technique but inverted. The following snippet denormalizes
data using an inverted point-relative normalization technique:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
def denormalize(reference, series, normalized_variable=\
                'close_point_relative_normalization', \
                denormalized_variable='close'):
    if('iso_week' in list(series.columns)):
        week_values = reference[reference['iso_week'] \
                      == series['iso_week'].values[0]]
        last_value = week_values[denormalized_variable].values[0]
        series[denormalized_variable] = \
        last_value * (series[normalized_variable] + 1)
    return series
predicted_close = predicted.groupby('iso_week').apply(lambda x: \
                  denormalize(observed, x))
```
:::

The `denormalize()`{.literal} function takes the first closing price
from the test\'s first day of an equivalent week.

Our results now compare the predicted values with the test set using US
dollars. As shown in the following plot, the `bitcoin_lstm_v0`{.literal}
model seems to perform quite well in predicting the Bitcoin prices for
the following 7 days. But how can we measure that performance in
interpretable terms?

<div>

::: {#_idContainer068 .IMG---Figure}
![Figure 3.8: De-normalized predictions per week
](2_files/B15911_03_08.jpg)
:::

</div>

Figure 3.8: De-normalized predictions per week

[]{#_idTextAnchor080}

Interpreting Predictions {#_idParaDest-78}
------------------------

Our last step is to add interpretability to our predictions. The
preceding plot seems to show that our model prediction matches the test
data somewhat closely, but how closely?

Keras\' `model.evaluate()`{.literal} function is useful for
understanding how a model is performing at each evaluation step.
However, given that we are typically using normalized datasets to train
neural networks, the metrics that are generated by the
`model.evaluate()`{.literal} method are also hard to interpret.

In order to solve that problem, we can collect the complete set of
predictions from our model and compare it with the test set using two
other functions from *Figure 3.1* that are easier to interpret: MAPE and
RMSE, which are implemented as `mape()`{.literal} and
`rmse()`{.literal}, respectively.

Note

These functions are implemented using NumPy. The original
implementations come from
<https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn>
and
<https://stackoverflow.com/questions/16774849/mean-squared-error-in-numpy>

We can see the implementation of these methods in the following snippet:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
def mape(A, B):
    return np.mean(np.abs((A - B) / A)) * 100
def rmse(A, B):
    return np.sqrt(np.square(np.subtract(A, B)).mean())
```
:::

After comparing our test set with our predictions using both of those
functions, we have the following results:

-   **Denormalized RMSE**: \$596.6 USD
-   **Denormalized MAPE**: 4.7 percent

This indicates that our predictions differ, on average, about \$596 from
real data. This represents a difference of about 4.7 percent from real
Bitcoin prices.

These results facilitate the understanding of our predictions. We will
continue to use the `model.evaluate()`{.literal} method to keep track of
how our LSTM model is improving, but will also compute both
`rmse()`{.literal} and `mape()`{.literal} on the complete series on
every version of our model to interpret how close we are to predicting
Bitcoin prices.

[]{#_idTextAnchor081}

Exercise 3.01: Creating an Active Training Environment {#_idParaDest-79}
------------------------------------------------------

In this exercise, we\'ll create a training environment for our neural
network that facilitates both its training and evaluation. This
environment is particularly important for the next topic, in which
we\'ll search for an optimal combination of hyperparameters.

First, we will start both a Jupyter Notebook instance and a TensorBoard
instance. Both of these instances can remain open for the remainder of
this exercise. Let\'s get started:

1.  Using your Terminal, navigate to the
    `Chapter03/Exercise3.01`{.literal} directory and execute the
    following code to start a Jupyter Notebook instance:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ jupyter-lab
    ```
    :::

    The server should open in your browser automatically.

2.  Open the Jupyter Notebook named
    `Exercise3.01_Creating_an_active_training_environment.ipynb`{.literal}:

    ::: {#_idContainer069 .IMG---Figure}
    ![Figure 3.9: Jupyter Notebook highlighting the section, Evaluate
    LSTM Model ](2_files/B15911_03_09.jpg)
    :::

    Figure 3.9: Jupyter Notebook highlighting the section, Evaluate LSTM
    Model

3.  Also using your Terminal, start a TensorBoard instance by executing
    the following command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ cd ./Chapter03/Exercise3.01/
    $ tensorboard --logdir=logs/
    ```
    :::

    Ensure the `logs`{.literal} directory is empty in the repository.

4.  Open the URL that appears on screen and leave that browser tab open,
    as well. Execute the initial cells containing the import statements
    to ensure that the dependencies are loaded.

5.  Execute the two cells under Validation Data to load the train and
    test datasets in the Jupyter Notebook:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    train = pd.read_csv('data/train_dataset.csv')
    test = pd.read_csv('data/test_dataset.csv')
    ```
    :::

    Note

    Don\'t forget to change the path (highlighted) of the files based on
    where they are saved on your system.

6.  Add TensorBoard callback and retrain the model. Execute the cells
    under Re-Train model with TensorBoard.

    Now, let\'s evaluate how our model performed against the test data.
    Our model is trained using 186 weeks to predict a week into the
    future---that is, the following sequence of 7 days. When we built
    our first model, we divided our original dataset between a training
    and a test set. Now, we will take a combined version of both
    datasets (let\'s call it a combined set) and move a sliding window
    of 186 weeks. At each window, we execute Keras\'
    `model.evaluate()`{.literal} method to evaluate how the network
    performed on that specific week.

7.  Execute the cells under the header, `Evaluate LSTM Model`{.literal}.
    The key concept of these cells is to call the
    `model.evaluate()`{.literal} method for each of the weeks in the
    test set. This line is the most important:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    result = model.evaluate(x=X_test, y=Y_test, verbose=0)
    ```
    :::

8.  Each evaluation result is now stored in the
    `evaluated_weeks`{.literal} variable. That variable is a simple
    array containing the sequence of MSE predictions for every week in
    the test set. Go ahead and plot the results:

    ::: {#_idContainer070 .IMG---Figure}
    ![Figure 3.10: MSE results from the model.evaluate() method for each
    week of the test set ](2_files/B15911_03_10.jpg)
    :::

    Figure 3.10: MSE results from the model.evaluate() method for each
    week of the test set

    As we\'ve already discussed, the MSE loss function is difficult to
    interpret. To facilitate our understanding of how our model is
    performing, we also call the `model.predict()`{.literal} method on
    each week from the test set and compare its predicted results with
    the set\'s values.

9.  Navigate to the *Interpreting Model Results* section and execute the
    code cells under the `Make Predictions`{.literal} subheading. Notice
    that we are calling the `model.predict()`{.literal} method, but with
    a slightly different combination of parameters. Instead of using
    both the `X`{.literal} and `Y`{.literal} values, we only use
    `X`{.literal}:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    predicted_weeks = []
    for i in range(0, test_data.shape[1]):
        input_series = combined_set[0:,i:i+186]
        predicted_weeks.append(model.predict(input_series))
    ```
    :::

    At each window, we will issue predictions for the following week and
    store the results. Now, we can plot the normalized results alongside
    the normalized values from the test set, as shown in the following
    plot:

    ::: {#_idContainer071 .IMG---Figure}
    ![Figure 3.11: Plotting the normalized values returned from
    model.predict() for each week of the test
    set](2_files/B15911_03_11.jpg)
    :::

    Figure 3.11: Plotting the normalized values returned from
    model.predict() for each week of the test set

    We will also make the same comparisons but using denormalized
    values. In order to denormalize our data, we must identify the
    equivalent week between the test set and the predictions. Then, we
    can take the first price value for that week and use it to reverse
    the point-relative normalization equation from *Chapter 2*,
    *Real-World Deep Learning with TensorFlow and Keras: Predicting the
    Price of Bitcoin*.

10. Navigate to the `De-normalized Predictions`{.literal} header and
    execute all the cells under that header.

11. In this section, we defined the `denormalize()`{.literal} function,
    which performs the complete denormalization process. In contrast to
    the other functions, this function takes in a pandas DataFrame
    instead of a NumPy array. We do this to use dates as an index. This
    is the most relevant cell block from that header:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    predicted_close = predicted.groupby('iso_week').apply(\
                      lambda x: denormalize(observed, x))
    ```
    :::

    Our denormalized results (as seen in the following plot) show that
    our model makes predictions that are close to the real Bitcoin
    prices. But how close?

    ::: {#_idContainer072 .IMG---Figure}
    ![Figure 3.12: Plotting the denormalized values returned from
    model.predict() for each week of the test set
    ](2_files/B15911_03_12.jpg)
    :::

    Figure 3.12: Plotting the denormalized values returned from
    model.predict() for each week of the test set

    The LSTM network uses MSE values as its loss function. However, as
    we\'ve already discussed, MSE values are difficult to interpret. To
    solve this, we need to implement two functions (loaded from the
    `utilities.py`{.literal} script) that implement the
    `rmse()`{.literal} and `mape()`{.literal} functions. These functions
    add interpretability to our model by returning a measurement on the
    same scale that our original data used, and by comparing the
    difference in scale as a percentage.

12. Navigate to the `De-normalizing Predictions`{.literal} header and
    load two functions from the `utilities.py`{.literal} script:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    from scripts.utilities import rmse, mape
    ```
    :::

    The functions from this script are actually really simple:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def mape(A, B):
      return np.mean(np.abs((A - B) / A)) * 100
    def rmse(A, B):
      return np.sqrt(np.square(np.subtract(A, B)).mean())
    ```
    :::

    Each function is implemented using NumPy\'s vector-wise operations.
    They work well in vectors of the same length. They are designed to
    be applied on a complete set of results.

    Using the `mape()`{.literal} function, we can now understand that
    our model predictions are about 4.7 percent away from the prices
    from the test set. This is equivalent to a RSME (calculated using
    the `rmse()`{.literal} function) of about \$596.6.

    Before moving on to the next section, go back into the Notebook and
    find the `Re-train Model with TensorBoard`{.literal} header. You may
    have noticed that we created a helper function called
    `train_model()`{.literal}. This function is a wrapper around our
    model that trains (using `model.fit()`{.literal}) our model, storing
    its respective results under a new directory. Those results are then
    used by TensorBoard in order to display statistics for different
    models.

13. Go ahead and modify some of the values for the parameters that were
    passed to the `model.fit()`{.literal} function (try epochs, for
    instance). Now, run the cells that load the model into memory from
    disk (this will replace your trained model):
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    model = load_model('bitcoin_lstm_v0.h5')
    ```
    :::

14. Now, run the `train_model()`{.literal} function again, but with
    different parameters, indicating a new run version. When you run
    this command, you will be able to train a newer version of the model
    and specify the newer version in the version parameter:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    train_model(model=model, X=X_train, Y=Y_train, \
                epochs=10, version=0, run_number=0)
    ```
    :::

    Note

    To access the source code for this specific section, please refer to
    <https://packt.live/2ZhK4z3>.

    You can also run this example online at
    <https://packt.live/2Dvd9i3>. You must execute the entire Notebook
    in order to get the desired result.

In this exercise, we learned how to evaluate a network using loss
functions. We learned that loss functions are key elements of neural
networks since they evaluate the performance of a network at each epoch
and are the starting point for the propagation of adjustments back into
layers and nodes. We also explored why some loss functions can be
difficult to interpret (for instance, the MSE function) and developed a
strategy using two other functions---RMSE and MAPE---to interpret the
predicted results from our LSTM model.

Most importantly, we\'ve concluded this exercise with an active training
environment. We now have a system that can train a deep learning model
and evaluate its results continuously. This will be key when we move on
to optimizing our network in the next topic.


Hyperparameter Optimization {#_idParaDest-80}
===========================

::: {#_idContainer081 .Content}
So far, we have trained a neural network to predict the next 7 days of
Bitcoin prices using the preceding 76 weeks of prices. On average, this
model issues predictions that are about 8.4 percent distant from real
Bitcoin prices.

This section describes common strategies for improving the performance
of neural network models:

-   Adding or removing layers and changing the number of nodes
-   Increasing or decreasing the number of training epochs
-   Experimenting with different activation functions
-   Using different regularization strategies

We will evaluate each modification using the same active learning
environment we developed by the end of the *Model Evaluation* section,
measuring how each one of these strategies may help us develop a more
precise model.

[]{#_idTextAnchor083}

Layers and Nodes -- Adding More Layers {#_idParaDest-81}
--------------------------------------

Neural networks with single hidden layers can perform fairly well on
many problems. Our first Bitcoin model (`bitcoin_lstm_v0`{.literal}) is
a good example: it can predict the next 7 days of Bitcoin prices (from
the test set) with error rates of about 8.4 percent using a single LSTM
layer. However, not all problems can be modeled with single layers.

The more complex the function you are working to predict, the higher the
likelihood that you will need to add more layers. A good way to
determine whether adding new layers is a good idea is to understand what
their role in a neural network is.

Each layer creates a model representation of its input data. Earlier
layers in the chain create lower-level representations, while later
layers create higher-level representations.

While this description may be difficult to translate into real-world
problems, its practical intuition is simple: when working with complex
functions that have different levels of representation, you may want to
experiment with adding layers.

[]{#_idTextAnchor084}

### Adding More Nodes {#_idParaDest-82}

The number of neurons that your layer requires is related to how both
the input and output data is structured. For instance, if you are
working on a binary classification problem to classify a 4 x 4 pixel
image, then you can try out the following:

-   Have a hidden layer that has 12 neurons (one for each available
    pixel)
-   Have an output layer that has only two neurons (one for each
    predicted class)

It is common to add new neurons alongside the addition of new layers.
Then, we can add a layer that has either the same number of neurons as
the previous one, or a multiple of the number of neurons from the
previous layer. For instance, if your first hidden layer has 12 neurons,
you can experiment with adding a second layer that has either 12, 6, or
24 neurons.

Adding layers and neurons can have significant performance limitations.
Feel free to experiment with adding layers and nodes. It is common to
start with a smaller network (that is, a network with a small number of
layers and neurons), then grow according to its performance gains.

If this comes across as imprecise, your intuition is right.

Note

To quote *Aurélien Géron*, YouTube\'s former lead for video
classification, \"*Finding the perfect amount of neurons is still
somewhat of a black art*.\"

Finally, a word of caution: the more layers you add, the more
hyperparameters you have to tune---and the longer your network will take
to train. If your model is performing fairly well and not overfitting
your data, experiment with the other strategies outlined in this chapter
before adding new layers to your network.

[]{#_idTextAnchor085}

Layers and Nodes -- Implementation {#_idParaDest-83}
----------------------------------

Now, we will modify our original LSTM model by adding more layers. In
LSTM models, we typically add LSTM layers in a sequence, making a chain
between LSTM layers. In our case, the new LSTM layer has the same number
of neurons as the original layer, so we don\'t have to configure that
parameter.

We will name the modified version of our model
`bitcoin_lstm_v1`{.literal}. It is good practice to name each one of the
models in terms of which one is attempting different hyperparameter
configurations. This helps you to keep track of how each different
architecture performs, and also to easily compare model differences in
TensorBoard. We will compare all the different modified architectures at
the end of this chapter.

Note

Before adding a new LSTM layer, we need to set the
`return_sequences`{.literal} parameter to `True`{.literal} on the first
LSTM layer. We do this because the first layer expects a sequence of
data with the same input as that of the first layer. When this parameter
is set to `False`{.literal}, the LSTM layer outputs the predicted
parameters in a different, incompatible output.

The following code example adds a second LSTM layer to the original
`bitcoin_lstm_v0`{.literal} model, making it
`bitcoin_lstm_v1`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
period_length = 7
number_of_periods = 186
batch_size = 1
model = Sequential() 
model.add(LSTM(
units=period_length,
batch_input_shape=(batch_size, number_of_periods, period_length), \
                  input_shape=(number_of_periods, period_length), \
                  return_sequences=True, stateful=False))
model.add(LSTM(units=period_length,\
               batch_input_shape=(batch_size, number_of_periods, \
                                  period_length), \
               input_shape=(number_of_periods, period_length), \
               return_sequences=False, stateful=False))
model.add(Dense(units=period_length)) \
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="rmsprop")
```
:::

[]{#_idTextAnchor086}

Epochs {#_idParaDest-84}
------

**Epochs** are the number of times the network adjusts its weights in
response to the data passing through and its resulting loss function.
Running a model for more epochs can allow it to learn more from data,
but you also run the risk of overfitting.

When training a model, prefer to increase the epochs exponentially until
the loss function starts to plateau. In the case of the
`bitcoin_lstm_v0`{.literal} model, its loss function plateaus at about
100 epochs.

Our LSTM model uses a small amount of data to train, so increasing the
number of epochs does not affect its performance in a significant way.
For instance, if we attempt to train it at 103 epochs, the model barely
gains any improvements. This will not be the case if the model being
trained uses enormous amounts of data. In those cases, a large number of
epochs is crucial to achieve good performance.

I suggest you use the following rule of thumb: *the larger the data used
to train your model, the more epochs it will need to achieve good
performance*.

[]{#_idTextAnchor087}

### Epochs -- Implementation {#_idParaDest-85}

Our Bitcoin dataset is rather small, so increasing the epochs that our
model trains may have only a marginal effect on its performance. In
order to have the model train for more epochs, we only have to change
the `epochs`{.literal} parameter in the `model.fit()`{.literal} method.
In the following snippet, you will see how to change the number of
epochs that our model trains for:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
number_of_epochs = 10**3 
model.fit(x=X, y=Y, batch_size=1,\
          epochs=number_of_epochs, \
          verbose=0, \
          callbacks=[tensorboard])
```
:::

This change bumps our model to `v2`{.literal}, effectively making it
`bitcoin_lstm_v2`{.literal}.

[]{#_idTextAnchor088}

Activation Functions {#_idParaDest-86}
--------------------

**Activation functions** evaluate how much you need to activate
individual neurons. They determine the value that each neuron will pass
to the next element of the network, using both the input from the
previous layer and the results from the loss function---or if a neuron
should pass any values at all.

Note

Activation functions are a topic of great interest for those in the
scientific community researching neural networks. For an overview of
research currently being done on the topic and a more detailed review on
how activation functions work, please refer to *Deep Learning by Ian
Goodfellow et. al., MIT Press, 2017*.

TensorFlow and Keras provide many activation functions---and new ones
are occasionally added. As an introduction, three are important to
consider; let\'s explore each of them.

Note

This section has been greatly inspired by the article *Understanding
Activation Functions in Neural Networks* by Avinash Sharma V, available
at
<https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0>.

[]{#_idTextAnchor089}

Linear (Identity) Functions {#_idParaDest-87}
---------------------------

Linear functions only activate a neuron based on a constant value. They
are defined by the following equation:

<div>

::: {#_idContainer073 .IMG---Figure}
![Figure 3.13: Formula for linear functions ](3_files/B15911_03_13.jpg)
:::

</div>

Figure 3.13: Formula for linear functions

Here, *c* is the constant value. When *c = 1*, neurons will pass the
values as is, without any modification needed by the activation
function. The issue with using linear functions is that, due to the fact
that neurons are activated linearly, chained layers now function as a
single large layer. In other words, we lose the ability to construct
networks with many layers, in which the output of one influences the
other:

<div>

::: {#_idContainer074 .IMG---Figure}
![Figure 3.14: Illustration of a linear function
](3_files/B15911_03_14.jpg)
:::

</div>

Figure 3.14: Illustration of a linear function

The use of linear functions is generally considered obsolete for most
networks because they do not compute complex features and do not induce
proper non-linearity in neurons.

[]{#_idTextAnchor090}

### Hyperbolic Tangent (Tanh) Function {#_idParaDest-88}

Tanh is a non-linear function, and is represented by the following
formula:

<div>

::: {#_idContainer075 .IMG---Figure}
![Figure 3.15: Formula for hyperbolic tangent function
](3_files/B15911_03_15.jpg)
:::

</div>

Figure 3.15: Formula for hyperbolic tangent function

This means that the effect they have on nodes is evaluated continuously.
Also, because of its non-linearity, we can use this function to change
how one layer influences the next layer in the chain. When using
non-linear functions, layers activate neurons in different ways, making
it easier to learn different representations from data. However, they
have a sigmoid-like pattern that penalizes extreme node values
repeatedly, causing a problem called vanishing gradients. Vanishing
gradients have negative effects on the ability of a network to learn:

<div>

::: {#_idContainer076 .IMG---Figure}
![Figure 3.16: Illustration of a tanh function
](3_files/B15911_03_16.jpg)
:::

</div>

Figure 3.16: Illustration of a tanh function

Tanhs are popular choices, but due to the fact that they are
computationally expensive, ReLUs are often used instead.

[]{#_idTextAnchor091}

### Rectified Linear Unit Functions {#_idParaDest-89}

**ReLU** stands for **Rectified Linear Unit**. It filters out negative
values and keeps only the positive values. ReLU functions are often
recommended as great starting points before trying other functions. They
are defined by the following formula:

<div>

::: {#_idContainer077 .IMG---Figure}
![Figure 3.17: Formula for ReLU functions ](3_files/B15911_03_17.jpg)
:::

</div>

Figure 3.17: Formula for ReLU functions

ReLUs have non-linear properties:

<div>

::: {#_idContainer078 .IMG---Figure}
![Figure 3.18: Illustration of a ReLU function
](3_files/B15911_03_18.jpg)
:::

</div>

Figure 3.18: Illustration of a ReLU function

ReLUs tend to penalize negative values. So, if the input data (for
instance, normalized between -1 and 1) contains negative values, those
will now be penalized by ReLUs. That may not be the intended behavior.

We will not be using ReLU functions in our network because our
normalization process creates many negative values, yielding a much
slower learning model.

[]{#_idTextAnchor092}

Activation Functions -- Implementation {#_idParaDest-90}
--------------------------------------

The easiest way to implement activation functions in Keras is by
instantiating the `Activation()`{.literal} class and adding it to the
`Sequential()`{.literal} model. `Activation()`{.literal} can be
instantiated with any activation function available in Keras (for a
complete list, see <https://keras.io/activations/>).

In our case, we will use the `tanh`{.literal} function. After
implementing an activation function, we bump the version of our model to
`v2`{.literal}, making it `bitcoin_lstm_v3`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model = Sequential() model.add(LSTM(
                               units=period_length,\
                               batch_input_shape=(batch_size, \
                               number_of_periods, period_length), \
                               input_shape=(number_of_periods, \
                                            period_length), \
                               return_sequences=True, \
                               stateful=False))
model.add(LSTM(units=period_length,\
          batch_input_shape=(batch_size, number_of_periods, \
                             period_length), \
          input_shape=(number_of_periods, period_length), \
          return_sequences=False, stateful=False))
model.add(Dense(units=period_length)) \
model.add(Activation("tanh"))
model.compile(loss="mse", optimizer="rmsprop")
```
:::

After executing the `compile`{.literal} command, your model has been
built according to the layers specified and is now ready to be trained.
There are a number of other activation functions worth experimenting
with. Both TensorFlow and Keras provide a list of implemented functions
in their respective official documentations. Before implementing your
own, start with the ones we\'ve already implemented in both TensorFlow
and Keras.

[]{#_idTextAnchor093}

Regularization Strategies {#_idParaDest-91}
-------------------------

Neural networks are particularly prone to overfitting. Overfitting
happens when a network learns the patterns of the training data but is
unable to find generalizable patterns that can also be applied to the
test data.

Regularization strategies refer to techniques that deal with the problem
of overfitting by adjusting how the network learns. In the following
sections, we\'ll discuss two common strategies:

-   L2 Regularization
-   Dropout

[]{#_idTextAnchor094}

### L2 Regularization {#_idParaDest-92}

**L2 regularization** (or **weight decay**) is a common technique for
dealing with overfitting models. In some models, certain parameters vary
in great magnitudes. L2 regularization penalizes such parameters,
reducing the effect of these parameters on the network.

L2 regularizations use the ![3](3_files/B15911_03_Formula_01.png)
parameter to determine how much to penalize a model neuron. We typically
set that to a very low value (that is, 0.0001); otherwise, we risk
eliminating the input from a given neuron completely.

[]{#_idTextAnchor095}

### Dropout {#_idParaDest-93}

Dropout is a regularization technique based on a simple question: *if we
randomly take away a proportion of the nodes from the layers, how will
the other node adapt?* It turns out that the remaining neurons adapt,
learning to represent patterns that were previously handled by those
neurons that are missing.

The dropout strategy is simple to implement and is typically very
effective at avoiding overfitting. This will be our preferred
regularization.

[]{#_idTextAnchor096}

### Regularization Strategies -- Implementation {#_idParaDest-94}

In order to implement the dropout strategy using Keras, we\'ll import
the `Dropout()`{.literal} method and add it to our network immediately
after each LSTM layer. This addition effectively makes our network
`bitcoin_lstm_v4`{.literal}. In this snippet, we\'re adding the
`Dropout()`{.literal} step to our model (`bitcoin_lstm_v3`{.literal}),
making it `bitcoin_lstm_v4`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model = Sequential()
model.add(LSTM(\
          units=period_length,\
          batch_input_shape=(batch_size, number_of_periods, \
                             period_length), \
          input_shape=(number_of_periods, period_length), \
          return_sequences=True, stateful=False))
model.add(Dropout(0.2))
model.add(LSTM(\
          units=period_length,\
          batch_input_shape=(batch_size, number_of_periods, \
                             period_length), \
          input_shape=(number_of_periods, period_length), \
          return_sequences=False, stateful=False))
model.add(Dropout(0.2))
model.add(Dense(units=period_length))
model.add(Activation("tanh"))
model.compile(loss="mse", optimizer="rmsprop")
```
:::

We could have used L2 regularization instead of dropout. Dropout drops
out random neurons in each epoch, whereas L2 regularization penalizes
neurons that have high weight values. In order to apply L2
regularization, simply instantiate the
`ActivityRegularization()`{.literal} class with the L2 parameter set to
a low value (for instance, 0.0001). Then, place it in the place where
the `Dropout()`{.literal} class has been added to the network. Feel free
to experiment by adding that to the network while keeping both
`Dropout()`{.literal} steps, or simply replace all the
`Dropout()`{.literal} instances with
`ActivityRegularization()`{.literal} instead.

[]{#_idTextAnchor097}

Optimization Results {#_idParaDest-95}
--------------------

All in all, we have created four versions of our model. Three of these
versions, that is, `bitcoin_lstm_v1`{.literal},
`bitcoin_lstm_v2`{.literal}, and `bitcoin_lstm_v3`{.literal}, were
created by applying different optimization techniques that were outlined
in this chapter. Now, we have to evaluate which model performs best. In
order to do that, we will use the same metrics we used in our first
model: MSE, RMSE, and MAPE. MSE is used to compare the error rates of
the model on each predicted week. RMSE and MAPE are computed to make the
model results easier to interpret. The following table shows this:

<div>

::: {#_idContainer080 .IMG---Figure}
![Figure 3.19: Model results for all models ](3_files/B15911_03_19.jpg)
:::

</div>

Figure 3.19: Model results for all models

Interestingly, our first model (`bitcoin_lstm_v0`{.literal}) performed
the best in nearly all defined metrics. We will be using that model to
build our web application and continuously predict Bitcoin prices.

[]{#_idTextAnchor098}

Activity 3.01: Optimizing a Deep Learning Model {#_idParaDest-96}
-----------------------------------------------

In this activity, we\'ll implement different optimization strategies on
the model we created in *Chapter 2*, *Real-World Deep Learning with
TensorFlow and Keras: Predicting the Price of Bitcoin*
(`bitcoin_lstm_v0`{.literal}). This model achieves a MAPE performance on
the complete de-normalization test set of about 8.4 percent. We will try
to reduce that gap and get more accurate predictions.

Here are the steps:

1.  Start TensorBoard from a Terminal.

2.  Start a Jupyter Notebook.

3.  Load the train and test data and split the `lstm`{.literal} input in
    the format required by the model.

4.  In the previous exercise, we create a model architecture. Copy that
    model architecture and add a new LSTM layer. Compile and create a
    model.

5.  Change the number of epochs in *step 4* by creating a new model.
    Compile and create a new model.

6.  Change the activation function to `tanh`{.literal} or
    `relu`{.literal} and create a new model. Compile and train a new
    model.

7.  Add a new layer for dropout after the LSTM layer and create a new
    model. Keep values such as `0.2`{.literal} or `0.3`{.literal} for
    dropout. Compile and train a new model.

8.  Evaluate and compare all the models that were trained in this
    activity.

    Note

    The solution to this activity can be found on page 141.


Summary {#_idParaDest-97}
=======

::: {#_idContainer081 .Content}
In this chapter, we learned how to evaluate our model using the MSE,
RMSE, and MAPE metrics. We computed the latter two metrics in a series
of 19-week predictions made by our first neural network model. By doing
this, we learned that it was performing well.

We also learned how to optimize a model. We looked at optimization
techniques, which are typically used to increase the performance of
neural networks. Also, we implemented a number of these techniques and
created a few more models to predict Bitcoin prices with different error
rates.

In the next chapter, we will be turning our model into a web application
that does two things: retrains our model periodically with new data and
is able to make predictions using an HTTP API interface.
