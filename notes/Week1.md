# ASL Machine Learning - Week 1

*Instructor:* Kyle Steckler

## Optimization

### Defining ML models

- ML models are functions with parameters and hyper-parameters;
- We will be focusing in supervised machine learning;
- Linear models have two types of parameters: Bias and weight;
  - In a hyperplane, $y = b + X \times w$, where $b$ is the Bias term, $X$ the input and $w$ the weight;
  - In linear, we define the equation by $y = b + x \times m$. The difference is only the dimensions of the model;
- Equation for a linear model tying mother's age and baby weight: $y=w_1x_1+b$, where $x_1$ is the feature (e.g. mother's age) and $w_1$ is the weight for $x_1$;

## Introducing Loss function

- Compose a loss function by calculating errors of the prediction;
- We use this function to measure the performance of each combination of the parameters;
- One loss function is Root Mean Squared Error: $ \sqrt{\frac{1}{n} \times \displaystyle\sum_{i=1}^{n}{(\hat{y_i}-y_i)^2}}$
- Logistic regression: Transform linear regression by a sigmoid activation function, represented by the $e^{-}$ part of the function;
  - $\hat{y} = \frac {1}{1+e^{-(w^Tx+b)}}$
- Typically, use cross-entropy (related to Shannon's information theory) as the erro metric;
  - $LogLoss=\displaystyle\sum_{(x,y) \in D}-y\log{(\hat{y})}-(1-y)\log{(1-\hat{y})}$
  - Less emphasis on errors where the output label

## Gradient descent

- Loss functions lead to loss surfaces;
  - We use this to find the minimal point of the surface, because it will be the set of parameters with minimal errors;
- To find the minimum we use the pseudocode bellow, where `espilon` is a tiny constant:

  ```python
  while loss > epsilon:
  direction = computeDirection()
  for i in range(params):
    params[i] = params[i] + stepSize * direction[i]
  loss = computeLoss()
  ```

- Small step sizes may never converge to the true minimum because it will to small changes over time. Large step sizes may never converge to the true minimum because the can overshoot it;
  - The ideal is to adapt the step size during the training. Start with larger step and reduce it over time;
- Complex models may have local minimals. Some times is hard to get to the global minima;
- The Loss Function slope provides de direction and step size in your search;
- So we can rewrite the pseudocode as follows:

  ```python
  while loss > epsilon:
  direction = computeDirection()
  for i in range(params):
    params[i] = params[i] + learning_rate * derivative[i]
  loss = computeLoss()
  ```

- Model training is still to slow;
  1. Calculate derivative: Number of data points or Number of model parameters;
  1. Take a step: Number of model parameters;
  1. Check Loss: Number of data points, number of model parameters, or frequency of checks;
- Calculating the derivative on fewer data points;
  - Creating subsets of data (batches) to train. This reduces the cost while preserving quality;
  - Data usually is chosen randomly, but is important that the algorithm sees all the samples;
  - The batch size should be representative as a whole;
- Checking loss with reduced frequency is a strategy to speed up the training process. For example: Update the loss every 100 steps or every hour;

### BigQuery and Data split

- Split a dataset into training/validation/test using hashing and modulo:

  ```sql
  SELECT *
  FROM bigquery
  WHERE MOD(ABS(FARM_FINGERPRINT(date)),10) < 8
  ```

  - Hash value on the date will always return the same value;
  - Then we can use a modulo operator to only pull 80% of the data based on the last few hash digits;
  - 10 is the amount of buckets that the data will be splitted;
- Carefully choose which field will split your data because it can't be used as a feature;
  - One technique is to hash on a JSON or a dummy column;
- In the development phase, prefer to use a small subset of data, because testing on the entire dataset ca be expensive;
- Baselines are important; It helps to know what error metric is reasonable or good;
- BQML is a way to easily build machine learning models using SQL;
- Models are similar to tables in BigQuery;
- To create a model in BigQuery, start by the query and add the model header:

  ```sql
  CREATE OR REPLACE MODEL
    dataset.model_name
  OPTIONS (
    input_label_cols=['label_column'],
    model_type='linear_reg'
  ) AS
  SELECT
    label_column,
    feature_column_1,
    feature_column_2
  FROM
    dataset.table
  ```

- Evaluate the model with `ML.EVALUATE`:

  ```sql
    SELECT
      *
    FROM
      ML.EVALUATE (
        MODEL dataset.model_name
      )
  ```

- If you don't set the data split, it will use 80/20 values as default;
- User the model with `ML.PREDICT`:

  ```sql
  SELECT
    *
  FROM
    ML.PREDICT(MODEL dataset.model_name, (
        SELECT
          feature_column_1,
          feature_column_2
        FROM
          dataset.table
    ))
  ```

- To create cross validation you need to do split by a separate query. It's on the roadmap of BQML;
- BigQuery handles categorical data using one-hot encoding automatically. But string are considered as a single value, not splitting the words;
- We can hide the `input_label_cols` in `OPTIONS` if the column with the label values is named `label`;
- FARM_FINGERPRINT returns an int64. We use ABS and MOD to create buckets. In this example, we are getting 1 on every 100.000 samples;
- During training we use mean squared error to avoid a square root operation in it. For analysis we apply the square root to bring the value back to the data scale;
- The machine learning algorithm represents about 5% of the solution. Most of the solution performance is about data exploration and cleaning, and its infrastructure and ops, and etc.;
- We can use `%%bigquery df` to get the results as Pandas DataFrames;
- [BigQuery CREATE MODEL documentation](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create);

## Core Tensorflow

### What is TensorFlow

- Open source, high performance library for numerical computation that uses directed graphs. It is not about just Machine Learning, but its core is a requirement for Machine Learning;
- Tensorflow works in DAGs. Graphs are defined by nodes and edges. The nodes represents operations, edges are the tensors;
- A tensor is a N-dimensional array of data;
  - Rank 0 - Scalar - Letter
  - Rank 1 - Vector - Row
  - Rank 2 - 2D Matrix - Page
  - Rank 3 - 3D Matrix - Book
  - Rank 4 - 4D Matrix - Book Shelf
  - Rank 5 - 5D Matrix - Library
- TensorFlow graphs are portable between different devices (CPUs, GPUs, TPUs, etc.);
- TensorFlow Lite provides on-device (edge) inference of ML models on mobile devices and is available for a variety of hardware;
  - It runs in iOS, Android, Raspberry Pi, etc.;
  - Training is done on the cloud;

### TensorFlow API hierarchy

- Contains multiple abstraction layers to support different hardwares;
- Composed by (lower to higher):
  - Core in C++ (deepest level, possible customization);
  - Core in Python (used to allow full control);
  - Components for NN models (tf.losses, tf.metrics, etc.);
  - High-level APIs for distributed training (tf.estimator, tf.keras, tf.data);
- Vertex AI has high integration with all those layers;

### Tensor and variable

- We can declare tensors using `tf.constant(data)`. The `tf.stack()` command allows to up a rank of a existing tensor;
  ```python
  x1 = tf.constant([2,3,4]) # (3,)
  x2 = tf.stack([x1,x1]) # (3,2,)
  ```
- A tensor is a N-dimensional array data;
- Tensors behave like numpy n-dimensional arrays, except that we have `tf.constant` and `tf.variable`;
- It is possible to slice a tensor just like an array;
- Tensor can be reshape using `tf.reshape(variable,shape)`;
- A tensor may receive a new value using

    ```python
    x = tf.Variable(2.0, dtype=tf.float32, name='my_variable')
    x.assign(48.2)    # 48.2
    x.assign_add(4)   # 52.2
    x.assign_sub(2)   # 50.2
    ```

### Autodiff and GradientTape

- GradientTape records operations for automatic derivative;
- GradientTape record the computation when it' executed (not when is defined!);

  ```python
  def compute_gradients(X, Y, w0, w1):
    with tf.GradientTape() as tape:
        loss = loss_mse(X, Y, w0, w1)
    return tape.gradient(loss, [w0, w1])    # Specify the function (loss) as well as the parameters you want to take the gradients with respect to ([w0, w1])

  w0 = tf.Variable(0.0)
  w1 = tf.Variable(0.0)

  dw0, dw1 = compute_gradients(X, Y, w0, w1)
  ```

- Autodiff eases the Gradient Descent implementation;

  ```python
  w0 = tf.Variable(0.0)
  w1 = tf.Variable(0.0)

  for step in range(0,STEPS + 1):
    dw0, dw1 = compute_gradient(X, Y, w0, w1)
    w0.assign_sub(dw0 * LEARNING_RATE)
    w1.assign_sub(dw1 * LEARNING_RATE)
  ```

## Training on Large Datasets

### The dataset API

- A tf.data.Dataset allow you to create data pipelines from in-memory dictionary and lists of tensors; or out-of-memory sharded data files;
- We can also preprocess data in parallel:

  ```python
    dataset = dataset.map(preproc_func).cache()
  ```

- Configure the way the data is fed into a model with a number of chaining methods

  ```python
    dataset = dataset.shuffle(1000)
                        .repeat(epochs)
                        .batch(batch_size)...
  ```

- We can think of `tf.data.Dataset` like generators, grabing batches of data at a time instead of downloading the whole data;
- Datasets can be created for different file formats, like `TextLineDataset`, `TFRecordDataset`, `FixedLengthRecordDataset`. There is also a `GenericDataset` that we can use to implement our own parser;
- For reading a csv file, we use `TextLineDataset`, then we define a parsing function to decode the data. TensorFlow provides the `tf.decode_csv` that makes it easier to do this;
- For sharded CSV files, we could use this:

  ```python
    dataset = tf.data.Dataset.list_files(path)
                    .flat_map(tf.data.TextLineDataset)
                    .map(parse_row)
  ```

- `tf.data.experimental.make_csv_dataset` function does something similar to the previous code;
- Separate the data for training, validation and testing before reading it;
- Prefetch allows the code to parallelize the reading and the usage. While one thread is reading and processing the data, another one can use it to analyze and train. It works in CPU (reading data) and GPU (processing);
- When `repeat` is empty, the default is `None`, which means $\infty$. So we need another mechanism to interrupt the training;

## Activation functions

- The composition of linear functions always collapses to a linear function. In order to Neural Networks to work, we need to add a Non-Linear function (aka Activation Function) to avoid collapsing to a linear function;
- We add the activation function between the hidden layers;
- The end of every neural network is always a linear or a logistic regression;
- A Neuron system is composed by the weighted sum, a Sigmoid function and hidden inputs;

> "Any math function can be approximated through a neural net (Universal Approximation Theorem)";

- Rectified Linear Unit (ReLU) is non-linear activation function widely used: $f(x)=\max(0,x)$;
- There are many different ReLU variants. One of them is the Parametric ReLU applies an $\alpha$ to values lower than 0 to avoid derivative issues. Another one is the ReLU6, which caps the values over 6 to avoid exploding values;

## Keras Sequential API

- Keras is now built-in to TF 2.x;
- Keras Sequential models bases on a sequences of layers;

  ```python
  model = Sequential([
    Input(shape=(64,)),
    Dense(units=32, activation="relu", name="hidden1")
    Dense(units=16, activation="relu", name="hidden2")
    Dense(units=8, activation="softmax", name="output")
  ])
  ```

- We can use `tf.keras.layers.Flatten` to flatten a matrix into an array;
- The last layer size is always equals to the number of classes desired with a activation function that helps the classification;
- After defining the model, we need to compile it using `model.compile()`;
- Adam is a optimizer very similar to Gradient Descent, with enhancements to solve some challenges of the original algorithm, like momentum that helps it to avoid being stuck in a local minima;
- We can implement our own metrics functions to use in the compile. The limitation is that this metric function has to use TFs functions;
- Once compiled, we call `model.fit` to start the training;
- Instead of defining how many epochs, we define the total number of training examples. This prevents that the algorithm reads 2 times one sample while reads another one only 1 time;

  ```python
    steps_per_epoch = NUM_TRAIN_EXAMPLES // (TRAIN_BATCH_SIZE * NUM_EVALS)

    history = model.fit(
        x=trainds,
        steps_per_epoch=steps_per_epoch,
        epochs = NUM_EVALS,
        validation_data=evalds,
        callbacks=[TensorBoard(LOGDIR)]
    )
  ```

- We use `sparse_categorical_crossentropy` when the values are integer. If we have one-hot encoding, we use `categorical_crossentropy`;
- Once trained, the model can be used for prediction using `model.predict(input_samples, steps=1)`;
- We can also save models using SavedModel, which is the universal serialization format for Tensorflow. This is an old version of the `model.save`, which is the recommended way of doing it;
- [Pre-built containers for Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers). These are used for serving predictions;
- In Vertex AI, we can setup two predicion job models:
  - For low-latency prediction, we use Endpoints;
  - For large predictions, like processing data collected during the day, we use Batch Processing;
- We can create a model in Vertex AI using `model.upload`. To create an endpoint in Vertex AI, we use `model.deploy` which will create an endpoint and deploy the model;
- We can query the endpoint through REST, gRPC or using a Client Library calling something like `endpoint.predict`. The Client Libraries are based on gRPC;

## Data preprocessing

- We can build and export end-to-end models that accept raw data as input. Models handle feature normalization or feature value indexing on their own;
- Keras has several preprocessing layers, like text, numerical, categorical and etc.;
- Exact floats are not meaningful. Think of a house location. The coordinates may be so meaningful as the city or neighborhood name. In this case we use buckets, that is create a group of values into a name, converting a discrete value to a categorical one;

  ```python
  latbuckets = np.linspace(start=38.0, stop=42.0, num=NBUCKETS).tolist()
  tf.keras.layers.Discretization(lonbuckets)(...)
  ```

  - The number of buckets is a hyper parameter that we should tune for each case;
- The discretization layer may generate an one-hot encoded vector or a integer index;
- Feature Crosses is essentially multiplying two features together. Ex.: Given the hour of a day (23 zeros and 1 one) and day of week (6 zeros and 1 one), we could combine them to create a new feature representing each combination of them (167 zeros and 1 one);
- [Tensorflow Playground](https://playground.tensorflow.org);
- HashedCrossing layers are the way to create a Feature Cross;
- Embedding columns are used to create a simpler representation of a more complex data;
- The model learns how to embed the feature cross in lower-dimensional space;
  - Example: Think of a traffic at 8 and 9 AM. They should be similar;
- Embedding represents data as lower-dimensional, dense vector;

  ```python
  embed = Embedding(input_dim=168, output_dim=2, name="pd_embed")(fc)
  ```

  - The goal of this is to create a numerical representation of something complex, like words. The result would be a one-hot encoded vector with the size of the set of possible values;
- Embedding layers allows us to convert a categorical index into a embed dense vector. Example: $[43] \rightarrow [4.9, 3.8, 0.28]$;
  - The goal is to go from high dimensionality to smaller dimensionalities;
- A good starting point for number of embedding dimensions: $dimensions \approx \sqrt[4]{possible\ values}$

## DNNs with the Keras Functional API

- As humans we have the ability to generalization (animals with wings can fly) and memorization (but penguins can't fly);
- Linear models are good for memorization, recommended for sparse and independent features. Ex.: words;
- Neural networks are good for generalization, recommended for dense, highly correlated features. Ex.: words in a phrase;
- Functional API allows to create wide and deep networks, with different inputs at different parallel layers;
- The functional API allows us to define the input of each one of the layers instead of a sequence. We can call the `Dense` object as a function passing the layer object as its input;
- Strengths:
  - Less verbose tha keras.Model;
  - Validates your model while you're defining it;
  - your model is plottable and inspectable;
  - Your model can be serialized or cloned;
- Weakness:
  - Doesn't support dynamic architectures;

## Advanced Feature Engineering with Keras

- We can create more meaningful features for the analysis. In our example we can calculate the euclidean distance between the pickup and drop off;
- To create this features, we use Lambda Layers which takes a Python function;

## Advanced Feature Engineering with BQML

- BQML has several preprocessing function like we saw with Keras;
- It has spatial and temporal functions out of the box, but we can also create our own;
- More complex data transform can be added to a `TRANSFORM` block;
- We must beware of overfitting as we increase the model complexity;
- In Gradient Descent we aim to minimize loss(Data|Model) + complexity (Model). The more complex the model is, more likely we will overfit it;
- One way to measure the model is by analyzing the weight vector. L1 regularization, complexity of model is defined by the L1 norm of the weight vector;
  - $L(w,D)+\lambda {\lvert\lvert w \lvert\lvert}_1$
- In L2 regularization **CHECK LATER**
- FeatureCross creates a one-hot encoding, while Concatenate literally concatenates the values. The size of both should be the same, it just changes the way the data is represented;

## Training on Vertex AI

-

## Hyperparameters tuning

- Model improvements is very sensitive to batch_size and learning_rate;
- The cost of usage of Vertex AI depends on the time of resource usage. So it is important to optimize this values;
- [Google Vizier](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf) is a service for black-box optimization. We can use it to optimize our hyperparameters. It learns from a previous set of parameters to do another trial;
- To use it we need to:
  1. The parameters must be command-line arguments;
  1. Set up `cloudml-hypertune` to record training metrics;
  1. Create a `StudySpec` for the hyperparameter `config.yaml` file;
  1. Create a `TrialJobSpec` for the hyperparameter `configu.yaml` file;
- The optimization trials run in parallel. It is possible to create some kind of sequence, running N jobs in parallel, analyze its results and start a new job batch based in these values;
