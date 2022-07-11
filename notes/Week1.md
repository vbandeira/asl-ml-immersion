# ASL Machine Learning - Week 1

*Instrutor:* Kyle Steckler

## Optimization

### Defining ML models

- ML models are functions with parameters and hyper-parameters;
- We will be focusing in supervised machine learning;
- Linear models have two types of parameters: Bias and weight;
  - In a hyperplane, $y = b + X \times w$ where $b$ is the Bias term, $X$ the input and $w$ the weight;
  - In linear, we define the equation by $y = b + x \times m$. The difference is only the dimensions of the model;
- Equation for a linear model tying mother's age and baby weight: $y=w_1x_1+b$, where $x_1$ is the feature (e.g. mother's age) and $w_1$ is the weight for $x_1$;

## Introducing Loss function

- Compose a loss function by calculating errors of the prediction;
- We use this function to measure the performance of each combination of the parameters;
- One loss function is Root Mean Squared Error: $ \sqrt{\frac{1}{n} \times \displaystyle\sum_{i=1}^{n}{(y_i-y_i)^2}}$
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
- Complex models may have local minimas. Some times is hard to get to the global minima;
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
  1. Check Loss: Number of data points, number of model parameteres, or frequency of checks;
- Calculating the derivative on fewer data points;
  - Creating subsets of data (batches) to train. This reduces the cost while preserving quality;
  - Data usually is chosen randomly, but is important that the algorithm sees all the samples;
  - The batch size should be representative as a whole;
- Checking loss with reduced frequency is a straetgy to speed up the training process. For example: Update the loss every 100 steps or every hour;

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
- To create a modelo in BigQuery, start by the query and add the model header:

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

## Tensor and variable

-
