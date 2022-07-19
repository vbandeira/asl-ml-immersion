# ASL Machine Learning - Week 2

*Instructor:* Kyle Steckler

## Review

- Its usual to have the configuration of feature and label columns in a separate Python file;
- The Hypertunning setup on `5a_train_keras_vertex_babyweight.ipynb` does 4 batches of 5 analysis. Each batch learns with the previous one;
- Usually we get the values from hyper tunning and use it in longer training runs or another longer hyper parameter tunning session. For example: You do the tuning using 10K steps. Then test the result in 50K steps;
- We can deploy the models in Vertex AI for usage by clients. The service auto scales based on the demand;
- We can use Cloud Run to do the predictions, but if it scales to zero, it will require a cold start. Once it starts, the rest of the requests are really fast;
- It is possible to deploy a prediction job without deploying an endpoint. This is because we can request batch predictions;
- Adding layers allows the network to analyze more complex data;

## Image Classification

### Linear Models for Image Classification

- MNIST is the "hello world" for image classification. It is composed by digits images, each with a size of 28x28 pixels;
- Images are arrays of numbers. In case of MINST is an array with single values between 0 and 255. Usually we scale it between 0 and 1;
- We flatten the images to have a single vector, concatenating each row after the other;
- $\Large softmax(x)_i= \frac{exp(x_i)} {\sum_j{exp(x_j)}}$
- $\frac{-1}{N} \times$...
- The simple way to solve this is to flatten the image, send it to a `Dense` layer and apply Softmax;
- The results of the classification is the probability of that image matches to each supported category;

### DNNs

- The Universal Approximation Theorem sets that any problem can be solved using linear functions;
- An infinitely large DNN could memorize anything;
- One of the best ways we have of mitigating overfitting is through the use of regularization;
- L1 regularization term: $\large\lambda\displaystyle\sum^k_{i=1}{\vert w_i \vert}$
- L2 regularization term: $\large\lambda\displaystyle\sum^k_{i=1}{w^2_i}$
- Dropout layers are a form of regularization, dropping a percentage of neuron links during training. We must be aware of vanishing gradient issues caused by this;
  - It reduces the dependency of the result to a specific neuron. Another way of thinking is that it adds noise to its hidden units. [Source](https://www.researchgate.net/profile/Satyanarayana-Vusirikala-2/post/What-is-the-difference-between-dropout-method-and-adding-noise-in-case-of-autoencoder/attachment/59d61eb46cda7b8083a1800a/AS%3A273572570828804%401442236191362/download/dropout+in+neural+networks.pdf);
- Usually we start with a dropout rate as 0.2;

## Intro to CNNs

- Using DNN for image classification coudl have billions of weight. Think of a 8 Megapixel image with 3 color channels;
- Traditionally, image classification tasks utilize a feature engineering step, like Harr Cascades, to feed your ML model;
  - Blurring algorithm kinds of calculate an average of each pixel and apply to the pixel;
  - People use techniques like this (and enhancing contrast, and etc) to create matrices that, when multiplied to the image, would return features for training;
- The idea of CNNs if to learn the matrices (image transformations) for the training;
- CNN feature engineering has two steps:
  - Convolution layers which learns the features for training;
  - Pooling, which reduces the dimension;
- A convolution is an operation that processes groups of nearby pixels, doing a sliding dot product, a matrix sum of weight multiplication to the input;
- A convolved feature map is created by multiplying the kernel weights by the features;
  - The weights of the kernel is shared through the whole image;
- Most convolutional neural networks commonly use multiple kernels;
- GPUs are great for multiplying matrices because they're built for it. TPUs were created based on this too;
- Different weight detect different features;
  - Example: Detects horizontal edges:
    $\begin{bmatrix}
    1 & 2 & 1 \\
    \empty & \empty & \empty \\
    -1 & -2 & -1
    \end{bmatrix}$
  - If we rotate this, we detect vertical edges. If we sum both results, we have edges of the image;
- CNNs can learn a hierarchy of features;
  - Edges to textures to patterns to parts to objects;
- Convolutional layers are collection of filters, like edge detection or brightness detection;
- We can combine two kernels in a single layer. Example: We have an image of 300x300x3 and pass it to a convolutional layer with 2 kernels. The result would be 296x296x2, the third dimension is created for each kernel;
- Padding preserves the shape of the input after the convolution;
- Keras provides a high level API to set up a convolutional layer:

  ```python
    tf.keras.layers.Conv2D(filters, #number of filters, i.e. out_channels
                            kernel_size=3, #size of the kernel, e.g. 3 for a 3x3 kernel
                            padding='same', #maintain the same shape across the input and output
                            stride=2 #Refers to the step size
                          )
  ```

- Maxpooling also reduces dimensionality. It is usually combined with convolutional layers;
  - It takes the maximum value of the stride window and send it to the next layer;
  - It kind of act as a regularization. It is more removing signals than adding noise;
  - We can think of it as reducing dimensionality of images without impacting the performance;
- There is also Minpooling and Meanpooling;
- Usually we have pairs of Convolution and Pooling layers in sequence until we have the Dense layers that we know;
- Tensorflow provides an API for Pooling: `tf.keras.layers.MaxPool2D(pool_size=2, strides = 2)`;
- Successive convolution layers apply filters to increasing scales;
- Typically we use 3x3 or 4x4 kernels;
  - Smaller kernel sizes tends to have better results, but it is empirical;
  - One way of think of it is how related are the neighbor pixels;

## Dealing with Data Scarcity

- Data need grows with model complexity;
- In the MNIST in linear mode, the parameter count is defined by: $height \times width \times num\_classes_{weight} + num\_classes_{bias}$;
  - Each neuron also has its own bias term;
- In the convolutions is smaller than regular one, because it considers the $num\_filters * kernel\_size + {num\_filters}_{bias}$;
- Real-world models have even more parameters. Alexnet (2012) has 60 million parameters. RestNet (2015) has 25 million;
- The simpler the model, lower the resource consumption and less complexity as well, resulting in better results;
- If you don't have enough labeled data, we can do Data augmentation and Transfer learning;

### Data augmentation

- Let's use the iris dataset. We have a small set of data available. And we have small amount of points that are not classified;
- If you small amount of data or it is skewed, you can create data using oversampling;
  - You find clusters in your data space that are bold with the same label and use them to create synthetic data;
  - One approach is to sample data closer to the centroid of the cluster, to avoid create data from the edges;
  - This works well for structured data;
- Common image augmentation techniques used are reshaping, sharping, blurring, mirroring, rotating, cropping, and etc.;
  - Changing hue, brightness and contrast are other common techniques;
- Data augmentation requires care to avoid issues related to wrong labels;
- Information is often in the detail;
  - Sometimes color is informative, in other the orientation or small detail are. Be careful about these when generate new images;
- Don't make the problem harder than it needs to be;
  - In a scenario of text translation of a photo, doesn't make sense to invert the image;
- TensorFlow has several image functions that we can use to augment this kind of data;

### Transfer learning

- Transfer learning is like a shortcut for training;
- Predicting the categories of images in ImageNet remains an important benchmark in computer vision;
- It allows us to take some pre-trained networks and reuse it;
- The first layers are most task-general. The last ones are more task-specific;
- Typically we cut the source network after the convolutional layers and retrain the Fully connected layers;
- If you don't have enough data, it may be interesting to freeze the source model. For large labeled data, let the source model train;
