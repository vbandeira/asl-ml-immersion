# ASL Machine Learning - Week 3

*Instructor:* Kyle Steckler

## Kuberflow Pipelines on Vertex AI

- Airflow has some integrations with Vertex AI as well;
- It is about ML lifecycle;
- ML systems can easily build up technical debt;
- ML solutions are more complex than traditional systems. You have to worry about everything related to software engineer plus your data. That adds complexity for testing, deploying and etc.;
- We also need to be aware of the model decay;
- Phases of a ML project:
  - Discovery Phase;
  - Is developing a model for this use case feasible?;
  - Development phase;
  - Deployment phase;
- Kubeflow provides a standardized platform for building ML pipelines. It has integration with Vertex AI;
- Types of Kubeflow components we will look at:
  1. Pre-built components;
  1. Lightweight Python components - Implement the component code;
  1. Custom components - Implements the code, package it into a container, write the component description;

### Pre-built components

- Simpler usage. We just need to import them;
- Using AutoML within Vertex Pipeline can speed up things;
- At the end of the process, each component created by code is a YAML file at the [Kubeflow's Pipeline repository](https://github.com/kubeflow/pipelines);
- When we import the modules, it is reading the YAML file and creating the Docker from it;
- All three types of Kubeflow components results in containers created by the YAML files. What changes are the components used to create the container;
- `asl-ml-immersion/notebooks/kubeflow_pipelines/pipelines/solutions/pipeline_vertex/tuning_lightweight_component.py` - Example of full work;

## Explainable AI

- It is the ability to "debug" your AI code. It allows us to analyze what a AI is doing;
- It uses a set of techniques, like highlighting pixels (IG Attribution Mask Overlay);
- Explainable AI belongs to the responsible AI set. It is a superset of interpretable ML;
- The simpler the model, it will be easier to explain it. Complex models are almost black boxes;
- We can have Intrinsic explainability by documenting the model and keeping it simple. For explainability after training, we use post-hoc;
  - Post-hoc splits into local analysis, based on individual predictions, and entire model, which aggregates and rank the contribution of input variables for the model;
  - We can explanations that are model specific or model agnostics;
- Partial dependence plots (PDPs): Show the marginal effect one or two features have on the predicted outcome of a machine learning models. It plots the results in charts;
  - It is very intuitive and easy to implement, but it is complicated to analyze multiple features, some PDP do not show the feature distribution, and it assumes that the features are independent;
- Permutation Feature Importance: the goal is to assign the importance of each feature;
  - It random shuffle the value of a feature and recalculates the error. The variation of the error says how important it is;
  - When two features have interaction, the changes affects both. And if the permutation is repeated, the results might vary greatly;
  - It is not deterministic. Usually changes the value of a feature 3 times and take an average of the error variation;
- Shapley Values: Is the average marginal contribution of a feature value across all possible coalitions;
  - It is model agnostic;
  - It works always with single examples;
  - It will compare the contribution of a feature with all coalitions of the features;
  - SHAP is a library that helps this analysis, but it is a computational expensive;
  - More complex than others;

### Integrated Gradients (IG)

- They are model specific because they are based on gradients;
- Gradient-based attribution is to check how much the label changes based on the change of the input;

  $\LARGE x_i=x_i \frac{\partial y}{\partial x_i}$

- It gives us a better insight of what is going on with the model;
- We take and already trained model, select an image as the input and set a baseline (for example setting the $\alpha=0$ resulting in a black image). Then we start growing the value of $\alpha$ and check the scaled gradient change. If the gradient stop growing, then it has no value for us because it saturated;

  $\Large IG_i(\textrm{image})={\textrm{image}}_i \displaystyle\int^1_0 \nabla F_i(\alpha \cdot \textrm{image})d\alpha$
- One of the big issues is that it requires a baseline image;
  $\large\textrm{IntegratedGrads}_i(x)::=(x_i-x_i')\times \int^1_{\alpha=0} \frac{\partial F(x' + \alpha \times (x-x'))}{\partial x_i}$
- Think of an image of a bug. If we start with a black image, we might get the wrong features. If we start with a white image, it may perform better;
- XRAI is an improvement upon IG. It is based in segmentation;
  - It computes the gradients for a black baseline and white baseline. Then it sums both attributions and identify the most importante regions;
  - It is more human interpretable;
- For custom models analyzed by Vertex AI, we need to save our TF model on Cloud Storage, create a model signature for serving predictions, upload the model to Vertex Model Registry, and...;

## Recommender systems

### Content-Based filters

- [Content-based filtering can by used to generate movie recommendations for multiple users at a time];
- Genre as columns (feature), every row is a movie. We sum the values of each column and normalize the values;
  - The genres are one-hot encoded, with the possibility of having more than one True value. Ex.: Action and Sci-fi;
- To get the score we multiply the normalized sum of the user profile to the features of each movie. The result is an indicator of how much the user would like the movie;
- Another approach is to get the score that each user gave to a certain movie. If the user evaluated a movie, then we should not recommend it again. But that is not a rule, it depends on each scenario;
- **Pros:** Doesn't need information about other users. Can recommend niche items;
- **Cons:** Requires domain knowledge to hand-engineer features. Difficult to expand interests of user.

### Collaborative filtering

- It learns latent factors and can explore outside user's personal bubble. It allows us to expand the interest of users;
- We organize items by similarity in two dimensional embeddings;
- Embedding spaces:
  1. Each user and item is a d-dimensional point within an embedding space;
  1. Embeddings can be learned from data;
  1. We're compressing the data to find the best generalities...;
- The factorization splits this matrix into row and column factors that are essentially user and item embeddings;

  $\LARGE A \approx U \times V^T$ where $A$ is the original matrix, $U$ and $V^T$ are the result matrices;

- We are saving a lot of spaces with this;

  $\LARGE k(users + movies)$

  $\LARGE k < \frac{U \times V}{2(U+V)}$

- Collaborative filtering is usually carried out using matrix factorization;

  $\LARGE \min_{U,V} \displaystyle \sum_{(i,j)\in obj}(A_{ij}-U_iV_j)^2$

- Gradient descent doesn't perform well on this scenario. We solve this using Alternative Least Squares (ALS);
  - ALS can solve the factorization, is parallel, flexible, and easy to handle unobserved interaction pairs. The downside is that it works only with least squares;
- Weighted ALS (WALS) sets the unobserved values are zero and they receive a weight with low confidence;

  $\large \displaystyle \sum_{(i,j)\in obj}(A_{ij}-U_iV_j)^2 + w_0 \times \displaystyle \sum_{(i,j)\notin obj}(0-U_iV_j)^2$

- The ALS algorithm works by alternating between rows and columns to factorize the matrix;

  $\large u_i=(\sum_{r_{ij} \in r_{i*}} v_j v_j^T + \lambda I_k )^{-1} \sum_{r_{ij}\in r_{i*}} r_ij v_j$

  $\large v_i=(\sum_{r_{ij} \in r_{i*}} u_i u_i^T + \lambda I_k )^{-1} \sum_{r_{ij}\in r_{i*}} r_{ij} u_i$

- Collaborative filtering seems powerful, but it has its drawbacks. Like fresh items/users;
- The cold-start problem affects collaborative filtering methods. We can define some thresholds to define which approach to use (content or collaborative);
- YouTube uses two neural networks to recommend videos, it is called Hydra. It uses a candidate generation (NN looking at all items and user data) that receives millions of videos and outputs hundreds. Then a ranking algorithm (NN looking for user data, other candidate sources and video features) that receives the output of candidate generation and outputs dozens of videos;
- AutoML has kind of a [recommender system](https://cloud.google.com/retail/docs/create-models). Essentially you upload all your user and item data. You select the metrics used, type of signal (implicit or explicit) and etc. You have to do more than basic AutoML trainings. It is not super popular because it has an overhead;

## Notes

- [Students group](https://www.linkedin.com/groups/13700936/)
- [Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)
- [Google Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/configuring-explanations)
- [Introducing BigQuery Flex Slots](https://cloud.google.com/blog/products/data-analytics/introducing-bigquery-flex-slots)
- [AutoML Recommender System - Feature](https://cloud.google.com/recommendations)
