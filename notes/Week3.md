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
-
