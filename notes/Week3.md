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
  <br/><br/>
  - $\LARGE x_i=x_i \frac{\partial y}{\partial x_i}$ <br/>
  <br/>
- It gives us a better insight of what is going on with the model;
- We take and already trained model, select an image as the input and set a baseline (for example setting the $\alpha=0$ resulting in a black image). Then we start growing the value of $\alpha$ and check the scaled gradient change. If the gradient stop growing, then it has no value for us because it saturated;
    - $\Large IG_i(\textrm{image})={\textrm{image}}_i \displaystyle\int^1_0 \nabla F_i(\alpha \cdot \textrm{image})d\alpha$
- One of the big issues is that it requires a baseline image;
  - $\large\textrm{IntegratedGrads}_i(x)::=(x_i-x_i')\times \int^1_{\alpha=0} \frac{\partial F(x' + \alpha \times (x-x'))}{\partial x_i}$
- Think of an image of a bug. If we start with a black image, we might get the wrong features. If we start with a white image, it may perform better;
- XRAI is an improvement upon IG. It is based in segmentation;
  - It computes the gradients for a black baseline and white baseline. Then it sums both attributions and identify the most importante regions;
  - It is more human interpretable;
- For custom models analyzed by Vertex AI, we need to save our TF model on Cloud Storage, create a model signature for serving predictions, upload the model to Vertex Model Registry, and...;

## Notes

- https://www.linkedin.com/groups/13700936/
- http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
