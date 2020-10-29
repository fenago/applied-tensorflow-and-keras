
4. Productization {#_idParaDest-98}
=================

::: {#_idContainer084 .Content}
:::

::: {#_idContainer091 .Content}
[]{#_idTextAnchor101}[]{#_idTextAnchor102}Overview

In this chapter, you will handle new data and create a model that is
able to learn continuously from the patterns it is shown and help make
better predictions. We will use a web application as an example to show
how to deploy deep learning models not only because of the simplicity
and prevalence of web apps, but also the different possibilities they
provide, such as getting predictions on mobile using a web browser and
making REST APIs for users.


Introduction {#_idParaDest-99}
============

::: {#_idContainer091 .Content}
This chapter focuses on how to *productize* a deep learning model. We
use the word productize to define the creation of a software product
from a deep learning model that can be used by other people and
applications.

We are interested in models that use new data as and when it becomes
available, continuously learn patterns from new data, and consequently,
make better predictions. In this chapter, we will study two strategies
to deal with new data: one that retrains an existing model, and another
that creates a completely new model. Then, we implement the latter
strategy in our Bitcoin price prediction model so that it can
continuously predict new Bitcoin prices.

By the end of this chapter, we will be able to deploy a working web
application (with a functioning HTTP API) and modify it to our heart\'s
content.


Handling New Data {#_idParaDest-100}
=================

::: {#_idContainer091 .Content}
Models can be trained once using a set of data and can then be used to
make predictions. Such static models can be very useful, but it is often
the case that we want our model to continuously learn from new
data---and to continuously get better as it does so.

In this section, we will discuss two strategies of handling new data and
see how to implement them in Python.

[]{#_idTextAnchor105}

Separating Data and Model {#_idParaDest-101}
-------------------------

When building a deep learning application, the two most important areas
are data and model. From an architectural point of view, it is
recommended that these two areas be kept separate. We believe that is a
good suggestion because each of these areas includes functions
inherently separate from each other. Data is often required to be
collected, cleaned, organized, and normalized, whereas models need to be
trained, evaluated, and able to make predictions.

Following that suggestion, we will be using two different code bases to
help us build our web application: the Yahoo Finance API and
`Model()`{.literal}:

-   The Yahoo Finance API: The API can be installed by using
    `pip`{.literal} with the following command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    pip install yfinance
    ```
    :::

    After installation, we will be able to access all the historical
    data related to the finance domain.

-   `Model()`{.literal}: This class implements all the code we have
    written so far into a single class. It provides facilities for
    interacting with our previously trained models and allows us to make
    predictions using de-normalized data, which is much easier to
    understand. The `Model()`{.literal} class is our model component.

These two code bases are used extensively throughout our example
application and define the data and model components.

[]{#_idTextAnchor106}

The Data Component {#_idParaDest-102}
------------------

The Yahoo Finance API helps to retrieve and parse the historical data of
stocks. It contains one relevant method, `history()`{.literal}, which is
detailed in the following code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
import yfinance as yf
ticker =  yf.Ticker("BTC-USD")
historic_data = ticker.history(period='max')
```
:::

This `history()`{.literal} method collects data from the Yahoo Finance
website, parses it, and returns a pandas DataFrame that is ready to be
used by the `Model()`{.literal} class.

The Yahoo Finance API uses the parameter ticker to determine what
cryptocurrency to collect. The Yahoo Finance API has many other
cryptocurrencies available, including popular ones such as Ethereum and
Bitcoin Cash. Using the `ticker`{.literal} parameter, you can change the
cryptocurrency and train a different model apart from the Bitcoin model
created in this book.

[]{#_idTextAnchor107}

The Model Component {#_idParaDest-103}
-------------------

The `Model()`{.literal} class is where we implement the application\'s
model component. The `Model()`{.literal} class contains five methods
that implement all the different modeling topics from this book. They
are the following:

-   `build()`{.literal}: This method builds an LSTM model using
    TensorFlow. This method works as a simple wrapper for a manually
    created model.
-   `train()`{.literal}: This method trains the model using data that
    the class was instantiated with.
-   `evaluate()`{.literal}: This method evaluates the model using a set
    of loss functions.
-   `save()`{.literal}: This method saves the model locally as a file.
-   `predict()`{.literal}: This method makes and returns predictions
    based on an input sequence of observations ordered by week.

We will use these methods throughout this chapter to work, train,
evaluate, and issue predictions with our model. The `Model()`{.literal}
class is an example of how to wrap essential TensorFlow functions into a
web application. The preceding methods can be implemented almost exactly
as in previous chapters, but with enhanced interfaces. For example, the
`train()`{.literal} method implemented in the following code trains a
model available in `self.model`{.literal} using data from
`self.X`{.literal} and `self.Y`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
def train(self, data=None, epochs=300, verbose=0, batch_size=1): 
    self.train_history = self.model.fit(x=self.X, y=self.Y, \
                                        batch_size=batch_size, \
                                        epochs=epochs, \
                                        verbose=verbose, \
                                        shuffle=False)
    self.last_trained = datetime.now()\
    .strftime('%Y-%m-%d %H:%M:%S') 
    return self.train_history
```
:::

The general idea is that each of the processes from the Keras workflow
(build or design, train, evaluate, and predict) can easily be turned
into distinct parts of a program. In our case, we have made them into
methods that can be invoked from the `Model()`{.literal} class. This
organizes our program and provides a series of constraints (such as on
the model architecture or certain API parameters), which help us deploy
our model in a stable environment.

In the following sections, we will explore common strategies for dealing
with new data.

[]{#_idTextAnchor108}

Dealing with New Data {#_idParaDest-104}
---------------------

The core idea of machine learning models---neural networks included---is
that they can learn patterns from data. Imagine that a model was trained
with a certain dataset and it is now issuing predictions. Now, imagine
that new data is available. There are different strategies you can
employ so that a model can take advantage of the newly available data to
learn new patterns and improve its predictions. In this section, we will
discuss two strategies:

-   Retraining an old model
-   Training a new model

[]{#_idTextAnchor109}

### Retraining an Old Model {#_idParaDest-105}

In this strategy, we retrain an existing model with new data. Using this
strategy, you can continuously adjust the model parameters to adapt to
new phenomena. However, data used in later training periods might be
significantly different from earlier data. Such differences might cause
significant changes to the model parameters, such as making it learn new
patterns and forget old patterns. This phenomenon is generally referred
to as **catastrophic forgetting**.

Note

Catastrophic forgetting is a common phenomenon affecting neural
networks. Deep learning researchers have been trying to tackle this
problem for many years. DeepMind, a Google-owned deep learning research
group from the United Kingdom, has made notable advancements in finding
a solution. The article, *Overcoming* *Catastrophic Forgetting in Neural
Networks*, *James Kirkpatrick*, *et. al*. is a good reference for such
work, and is available at <https://arxiv.org/pdf/1612.00796.pdf>.

The interface used for training (`model.fit()`{.literal}) for the first
time can be used for training with new data as well. The following
snippet loads the data and helps to train a model specifying the epochs
and batch size:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
X_train_new, Y_train_new = load_new_data()
model.fit(x=X_train_new, y=Y_train_new, batch_size=1, \
          epochs=100, verbose=0)
```
:::

In TensorFlow, when models are trained, the model\'s state is saved as
weights on the disk. When you use the `model.save()`{.literal} method,
that state is also saved. And when you invoke the
`model.fit()`{.literal} method, the model is retrained with the new
dataset, using the previous state as a starting point.

In typical Keras models, this technique can be used without further
issues. However, when working with LSTM models, this technique has one
key limitation: the shape of both train and validation data must be the
same. For example, in *Chapter 3*, *Real-World Deep Learning with
TensorFlow and Keras: Evaluating the Bitcoin Model*, our LSTM model
(`bitcoin_lstm_v0`{.literal}) uses 186 weeks to predict one week into
the future. If we attempt to retrain the model with 187 weeks to predict
the coming week, the model raises an exception with information
regarding the incorrect shape of data.

One way of dealing with this is to arrange data in the format expected
by the model. For example, to make predictions based on a year\'s data
(52 weeks), we would need to configure a model to predict a future week
using 40 weeks. In this case, we first train the model with the first 40
weeks of 2019, then continue to retrain it over the following weeks
until we reach week 51. We use the `Model()`{.literal} class to
implement a retraining technique in the following code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
M = Model(data=model_data[0*7:7*40 + 7], variable='close', \
          predicted_period_size=7)
M.build()
M.train()
for i in range(41, 52):
    j = i - 40
    M.train(model_data.loc[j*7:7*i + 7])
```
:::

This technique tends to be fast to train and tends to work well with
series that are large. The next technique is easier to implement and
works well in smaller series.

[]{#_idTextAnchor110}

### Training a New Model {#_idParaDest-106}

Another strategy is to create and train a new model every time new data
is available. This approach tends to reduce catastrophic forgetting, but
training time increases as data increases. Its implementation is quite
simple.

Using the Bitcoin model as an example, let\'s now assume that we have
old data for 49 weeks of 2019, and that after a week, new data is
available. We represent this with the `old_data`{.literal} and
`new_data`{.literal} variables in the following snippet, in which we
implement a strategy for training a new model when new data is
available:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
old_data = model_data[0*7:7*48 + 7]
new_data = model_data[0*7:7*49 + 7]
M = Model(data=old_data,\
          variable='close', predicted_period_size=7)
M.build()
M.train()
M = Model(data=new_data,\
          variable='close', predicted_period_size=7)
M.build()
M.train()
```
:::

This approach is very simple to implement and tends to work well for
small datasets. This will be the preferred solution for our Bitcoin
price-predictions application.

[]{#_idTextAnchor111}

Exercise 4.01: Retraining a Model Dynamically {#_idParaDest-107}
---------------------------------------------

In this exercise, you have to retrain a model to make it dynamic.
Whenever new data is loaded, it should be able to make predictions
accordingly. Here are the steps to follow:

Start by importing `cryptonic`{.literal}. Cryptonic is a simple software
application developed for this book that implements all the steps up to
this section using Python classes and modules. Consider Cryptonic as a
template to be used to create applications. Cryptonic, provided as a
Python module for this exercise, can be found at the following GitHub
link: <https://packt.live/325WdZQ>.

1.  First, we will start a Jupyter Notebook instance, and then we will
    load the `cryptonic`{.literal} package.

2.  Using your Terminal, navigate to the
    `Chapter04/Exercise4.01`{.literal} directory and execute the
    following code to start a Jupyter Notebook instance:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ jupyter-lab
    ```
    :::

    The server will automatically open in your browser, then open the
    Jupyter Notebook named
    `Exercise4.01_Re_training_a_model_dynamically.ipynb`{.literal}.

3.  Now, we will import classes from the `cryptonic`{.literal} package:
    `Model()`{.literal} and the Yahoo Finance API. These classes
    facilitate the process of manipulating our model.

4.  In the Jupyter Notebook instance, navigate to the header
    `Fetching Real-Time Data`{.literal}. We will now be fetching updated
    historical data from Yahoo Finance by calling the
    `history()`{.literal} method:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    import yfinance as yf
    ticker =  yf.Ticker("BTC-USD")
    historic_data = ticker.history(period='max')
    ```
    :::

    The `historic_data`{.literal} variable is now populated with a
    pandas DataFrame that contains historic data of Bitcoin rates up to
    the time of running this code. This is great and makes it easier to
    retrain our model when more data is available.

5.  You can view the first three rows of data stored in
    `historic_data`{.literal} using the following command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    historic_data.head(3)
    ```
    :::

    You can then view this data stored in `historic_data`{.literal}:

    ::: {#_idContainer085 .IMG---Figure}
    ![Figure 4.1: Output displaying the head of the data
    ](3_files/B15911_04_01.jpg)
    :::

    Figure 4.1: Output displaying the head of the data

    The data contains practically the same variables from the Bitcoin
    dataset we used. However, much of the data comes from an earlier
    period, 2017 to 2019.

6.  Using the pandas API, filter the data for only the dates available
    in 2019, and store them in `model_data`{.literal}. You should be
    able to do this by using the date variable as the filtering index.
    Make sure the data is filtered before you continue:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    start_date = '01-01-2019'
    end_date = '31-12-2019'
    mask = ((historic_data['date'] \
             >= start_date) & (historic_data['date'] \
             <= end_date))
    model_data = historic_data[mask]
    ```
    :::

    Run `model_data`{.literal} in next cell and the output model can be
    seen as follows:

    ::: {#_idContainer086 .IMG---Figure}
    ![Figure 4.2: The model\_data variable showing historical data
    ](3_files/B15911_04_02.jpg)
    :::

    Figure 4.2: The model\_data variable showing historical data

    The `Model()`{.literal} class compiles all the code we have written
    so far in all of our activities. We will use that class to build,
    train, and evaluate our model in this activity.

7.  We will now use the filtered data to train the model:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    M = Model(data=model_data, \
              variable='close', predicted_period_size=7)
    M.build()
    M.train()
    M.predict(denormalized=True)
    ```
    :::

8.  Run the following command to see the trained model:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    M.train(epochs=100, verbose=1)
    ```
    :::

    The trained model is shown in the following screenshot:

    ::: {#_idContainer087 .IMG---Figure}
    ![Figure 4.3: The output showing our trained model
    ](3_files/B15911_04_03.jpg)
    :::

    Figure 4.3: The output showing our trained model

    The preceding steps showcase the complete workflow when using the
    `Model()`{.literal} class to train a model.

    Note

    For the complete code, use the `Chapter04/Exercise4.01`{.literal}
    folder.

9.  Next, we\'ll focus on retraining our model every time more data is
    available. This readjusts the weights of the network to new data.

    In order to do this, we have configured our model to predict a week
    using 40 weeks. We now want to use the remaining 11 full weeks to
    create overlapping periods of 40 weeks. These include one of those
    11 weeks at a time, and retrain the model for every one of those
    periods.

10. Navigate to the `Re-Train Old Model`{.literal} header in the Jupyter
    Notebook. Now, complete the `range`{.literal} function and the
    `model_data`{.literal} filtering parameters using an index to split
    the data into overlapping groups of seven days. Then, retrain our
    model and collect the results:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    results = []
    for i in range(A, B): 
        M.train(model_data[C:D])
        results.append(M.evaluate())
    ```
    :::

    The `A`{.literal}, `B`{.literal}, `C`{.literal}, and `D`{.literal}
    variables are placeholders. Use integers to create overlapping
    groups of seven days in which the overlap is of one day.

    Replacing these placeholders with weeks, we run the loop as follows:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    results = []
    for i in range(41, 52):
        j = i-40
        print("Training model {0} for week {1}".format(j,i))
        M.train(model_data.loc[j*7:7*i+7])
        results.append(M.evaluate())
    ```
    :::

    Here\'s the output showing the results of this loop:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    Training model 1 for week 41
    Training model 2 for week 42
    Training model 3 for week 43
    Training model 4 for week 44
    Training model 5 for week 45
    Training model 6 for week 46
    Training model 7 for week 47
    Training model 8 for week 48
    Training model 9 for week 49
    Training model 10 for week 50
    Training model 11 for week 51
    ```
    :::

11. After you have retrained your model, go ahead and invoke the
    `M.predict(denormalized=True)`{.literal} function and examine the
    results:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    array([7187.145 , 7143.798 , 7113.7324, 7173.985 , 7200.346 ,
           7300.2896, 7175.3203], dtype=float32)
    ```
    :::

12. Next, we\'ll focus on creating and training a new model every time
    new data is available. In order to do this, we now assume that we
    have old data for 49 weeks of 2019, and after a week, we now have
    new data. We represent this with the `old_data`{.literal} and
    `new_data`{.literal} variables.

13. Navigate to the `New Data New Model`{.literal} header and split the
    data between the `old_data`{.literal} and `new_data`{.literal}
    variables:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    old_data = model_data[0*7:7*48 + 7]
    new_data = model_data[0*7:7*49 + 7]
    ```
    :::

14. Then, train the model with `old_data`{.literal} first:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    M = Model(data=old_data,\
              variable='close', predicted_period_size=7)
    M.build()
    M.train()
    ```
    :::

    We now have all the pieces that we need in order to train our model
    dynamically.

    Note

    To access the source code for this specific section, please refer to
    <https://packt.live/2AQb3bE>.

    You can also run this example online at
    <https://packt.live/322KuLl>. You must execute the entire Notebook
    in order to get the desired result.

In the next section, we will deploy our model as a web application,
making its predictions available in the browser via an HTTP API.

[]{#_idTextAnchor112}

Deploying a Model as a Web Application {#_idParaDest-108}
--------------------------------------

In this section, we will deploy our model as a web application. We will
use the Cryptonic web application to deploy our model, exploring its
architecture so that we can make modifications in the future. The
intention is to have you use this application as a starter for more
complex applications---a starter that is fully working and can be
expanded as you see fit.

Aside from familiarity with Python, this topic assumes familiarity with
creating web applications. Specifically, we assume that you have some
knowledge of web servers, routing, the HTTP protocol, and caching. You
will be able to locally deploy the demonstrated Cryptonic application
without extensive knowledge of these web servers, the HTTP protocol, and
caching, but learning these topics will make any future development much
easier.

Finally, Docker is used to deploy our web applications, so basic
knowledge of that technology is also useful.

Before we continue, make sure that you have the following applications
installed and available on your computer:

-   Docker (Community Edition) 17.12.0-ce or later
-   Docker Compose (`docker-compose`{.literal}) 1.18.0 or later

Both these components can be downloaded and installed on all major
systems from <http://docker.com/>. These are essential for completing
this activity. Make sure these are available in your system before
moving forward.

[]{#_idTextAnchor113}

Application Architecture and Technologies {#_idParaDest-109}
-----------------------------------------

In order to deploy our web applications, we will use the tools and
technologies described in *Figure 4.4*. Flask is key because it helps us
create an HTTP interface for our model, allowing us to access an HTTP
endpoint (such as `/predict`{.literal}) and receive data back in a
universal format. The other components are used because they are popular
choices when developing web applications:

<div>

::: {#_idContainer088 .IMG---Figure}
![Figure 4.4: Tools and technologies used for deploying a deep learning
web application ](3_files/B15911_04_04.jpg)
:::

</div>

Figure 4.4: Tools and technologies used for deploying a deep learning
web application

These components fit together as shown in the following diagram:

<div>

::: {#_idContainer089 .IMG---Figure}
![Figure 4.5: System architecture for the web application built in this
project ](3_files/B15911_04_05.jpg)
:::

</div>

Figure 4.5: System architecture for the web application built in this
project

A user visits the web application using their browser. That traffic is
then routed by Nginx to the Docker container containing the Flask
application (by default, running on port `5000`{.literal}). The Flask
application has instantiated our Bitcoin model at startup. If a model
has been given, it uses that model without training; if not, it creates
a new model and trains it from scratch using data from Yahoo Finance.

After having a model ready, the application verifies if the request has
been cached on Redis; if yes, it returns the cached data. If no cache
exists, then it will go ahead and issue predictions, which are rendered
in the UI.

[]{#_idTextAnchor114}

Exercise 4.02: Deploying and Using Cryptonic {#_idParaDest-110}
--------------------------------------------

Cryptonic is developed as a dockerized application. In Docker terms,
this means that the application can be built as a Docker image and then
deployed as a Docker container in either a development or a production
environment.

In this exercise, we will see how to use Docker and Cryptonic to deploy
the application. Before you begin, download Docker for Desktop from
<https://www.docker.com/products/docker-desktop> Make sure that this
application is running in the background before beginning the exercise.

Note

The complete code for this exercise can be found at
<https://packt.live/2AM5mLP>.

1.  Docker uses files called `Dockerfiles`{.literal} to describe the
    rules for how to build an image and what happens when that image is
    deployed as a container. Cryptonic\'s Dockerfile is available in the
    following code:

    Note

    The triple-quotes ( `"""`{.literal} ) shown in the code snippet
    below are used to denote the start and end points of a multi-line
    code comment. Comments are added into code to help explain specific
    bits of logic.

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    FROM python:3.6 
    ENV TZ=America/New_York
    """
    Setting up timezone to EST (New York)
    Change this to whichever timezone your data is configured to use.
    """
    RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
    COPY . /cryptonic
    WORKDIR "/cryptonic"
    RUN pip install -r requirements.txt
    EXPOSE 5000
    CMD ["python", "run.py"]
    ```
    :::

2.  A Dockerfile can be used to build a Docker image with the following
    command:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ docker build --tag cryptonic:latest
    ```
    :::

    This command will make the `cryptonic:latest`{.literal} image
    available to be deployed as a container. The building process can be
    repeated on a production server, or the image can be directly
    deployed and then run as a container.

3.  After an image has been built and is available, you can run the
    Cryptonic application by using the `docker run`{.literal} command,
    as shown in the following code:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ docker run --publish 5000:5000 \
    --detach cryptonic:latest
    ```
    :::

    The `--publish`{.literal} flag binds port `5000`{.literal} on
    localhost to port `5000`{.literal} on the Docker container, and
    `--detach`{.literal} runs the container as a daemon in the
    background.

    In case you have trained a different model and would like to use
    that instead of training a new model, you can alter the
    `MODEL_NAME`{.literal} environment variable on the
    `docker-compose.yml`{.literal}. That variable should contain the
    filename of the model you have trained and want served (for example,
    `bitcoin_lstm_v1_trained.h5`{.literal}); it should also be a Keras
    model. If you do that, make sure to also mount a local directory
    into the `/models`{.literal} folder. The directory that you decide
    to mount must contain your model file.

    The Cryptonic application also includes several environment
    variables that you may find useful when deploying your own model:

    `MODEL_NAME`{.literal}: Allows us to provide a trained model to be
    used by the application.

    `BITCOIN_START_DATE`{.literal}: Determines which day to use as the
    starting day for the Bitcoin series. Bitcoin prices have a lot more
    variance in recent years than earlier ones. This parameter filters
    the data to only years of interest. The default is
    `January 1, 2017`{.literal}.

    `PERIOD_SIZE`{.literal}: Sets the period size in terms of days. The
    default is `7`{.literal}.

    `EPOCHS`{.literal}: Configures the number of epochs that the model
    trains on every run. The default is `300`{.literal}.

    These variables can be configured in the
    `docker-compose.yml`{.literal} file. A part of this file is shown in
    the following code snippet:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    version: "3" 
       services:
          cache:
             image: cryptonic-cache:latest
             build:
                context: ./cryptonic-cache
                dockerfile: ./Dockerfile
             volumes:
                - $PWD/cache_data:/data
             networks:
                - cryptonic
          cryptonic:
             image: cryptonic:latest
             build:
                context: .
                dockerfile: ./Dockerfile
             ports:
                - "5000:5000"
             environment:
                - BITCOIN_START_DATE=2019-01-01
                - EPOCH=50
                - PERIOD_SIZE=7
    ```
    :::

4.  The easiest way to deploy Cryptonic is to use the
    `docker-compose.yml`{.literal} file in the repository
    (<https://packt.live/2AM5mLP>).

    This file contains all the specifications necessary for the
    application to run, including instructions on how to connect with
    the Redis cache and what environment variables to use. After
    navigating to the location of the `docker-compose.yml`{.literal}
    file, Cryptonic can then be started with the
    `docker-compose up`{.literal} command, as shown in the following
    code:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    $ docker-compose up -d
    ```
    :::

    The `-d`{.literal} flag executes the application in the background.

5.  After deployment, Cryptonic can be accessed on port `5000`{.literal}
    via a web browser. The application has an HTTP API that makes
    predictions when invoked. The API has the endpoint
    `/predict`{.literal}, which returns a JSON object containing the
    de-normalized Bitcoin price prediction for a week into the future.
    Here\'s a snippet showing an example JSON output from the
    `/predict`{.literal} endpoint:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    {
      message: "API for making predictions.",
      period_length: 7,
        result: [ 15847.7,
          15289.36,
          17879.07,
          …
          17877.23,
          17773.08
        ],
        success: true,
        7
    }
    ```
    :::

    Note

    To access the source code for this specific section, please refer to
    <https://packt.live/2ZZlZMm>.

    This section does not currently have an online interactive example,
    and will need to be run locally.

The application can now be deployed on a remote server and you can then
use it to continuously predict Bitcoin prices. You\'ll be deploying an
application in the activity that follows.

[]{#_idTextAnchor115}

Activity 4.01: Deploying a Deep Learning Application {#_idParaDest-111}
----------------------------------------------------

In this section, based on the concepts explained up to now, try
deploying the model as a local web application. You will need to follow
these steps:

1.  Navigate to the `cryptonic`{.literal} directory.
2.  Build the Docker images for the required components.
3.  Change the necessary parameters in `docker-compose.yml`{.literal}.
4.  Deploy the application using Docker on the localhost.

The expected output would be as follows:

<div>

::: {#_idContainer090 .IMG---Figure}
![Figure 4.6: Expected output ](3_files/B15911_04_06.jpg)
:::

</div>

Figure 4.6: Expected output

Note

The solution for this activity can be found on page 150.


Summary {#_idParaDest-112}
=======

::: {#_idContainer091 .Content}
This lesson concludes our journey into creating a deep learning model
and deploying it as a web application. Our very last steps included
deploying a model that predicts Bitcoin prices built using Keras and the
TensorFlow engine. We finished our work by packaging the application as
a Docker container and deploying it so that others can consume the
predictions of our model, as well as other applications, via its API.

Aside from that work, you have also learned that there is much that can
be improved. Our Bitcoin model is only an example of what a model can do
(particularly LSTMs). The challenge now is twofold: how can you make
that model perform better as time passes? And what features can you add
to your web application to make your model more accessible? With the
concepts you\'ve learned in this book, you will be able to develop
models and keep enhancing them to make accurate predictions.
