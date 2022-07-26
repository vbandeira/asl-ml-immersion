# ASL - Reinforcement Learning

*Instructor:* Benoit Dherin

- In RL we have natural and manufactured labels, like creating word embeddings;
- We have:
  - Agent - An intelligence that can act and transition between states (stock trader);
  - State (s) - The environmental parameters describing where the agent is (last highs and lows);
  - Action (a) - What the agent can do within a state (go long or short);
  - Time/Step (t)- One transition between states (moving in time);
  - Reward R(s,a) - Value gained or lost transitioning between states through and action (R(Battle, Kill Monster)=100XP);
- At each time/step, the agent receives the previous state and the reward. The agent then think and decides what action it will take and send it to the environment and the process restart;
- The training goal: Learning the optimal policy;
- The return G is the total discounted reward from time-step t.\
$\large G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+...$
  - The $\gamma$ is the discount factor;
  - $\gamma \in [0,1]$
- Anything can be a state. The state is similar to a feature in traditional machine learning;
- The value is the expected return from a given state following a given policy. The Q-Value is the expected return from a given state and action following a given policy. Both values depend on the choice of a policy;
- TensorFlow as TF-Agents that we can use for this approach;
- The environment is an interface (`PyEnvironment`) that we can do our own implementations;
  - The environment contains the actions and observations;
- Reinforcement learning is more focused on:
  - Control/Optimization/Decision making problem;
  - Is mostly offline simulation or realtime training;
  - Trial and error necessary;
  - Transfer learning not yet possible;
  - Optimized for long term/delayed value oriented;
- Reinforcement learning is a superset of supervised learning. It is possible to convert supervised into reinforcement. Environment produces state/context x at time t. X would be the state and Y the actions;
- Contextual Bandits are RL environment for simple problems. They are used by Ads and recommender systems like Spotify, Amazon recommendations and the Google Play Store for app discovery;
- Another good example is NLP Text Summarization. The state is the `Source` + what has been summarized so far. The actions are the next possible words. The reward would be the cross-entropy of the word vectors. The reward is composed by value representing the fluency, its relevance and conciseness;
  - The composition of the rewards should be calculated automatically. We need to think how to build this features;
- Using and cooling example, the state could be the temperature, workloads, bill, energy consumption, etc. For the action we could use the workload distribution, air conditioning values, and etc. The rewards could be the energy economy, temperature, etc.;
- To create a RL to play classic games, we can use image from the game (processed by a CNN) and use this as the state. The actions could be all the actions possible in the game. The reward could be the progress in the game, score, survival time;
  - In some cases the image should be a short video to analyze the situation;
- We can use model-based or model-free approaches in RL algorithms. Model free are more simple and practical, but they require more simulation/live training;
- We can use a Value Based or a Policy Based. The policy is based in the probability of the best action without estimating the action value. The value based is more deterministic;
  - Value Based: $\large Q_{\theta}(s,a)$
  - Policy Based: $\large\pi_{\theta}(s,a)$

## Value based algorithms

- The key point is how to create labels based on the history. With this we can apply gradient descent;
- The Q-Value is the expected value after applying an action based on a state;
- For each time step, we collect an $s$ and a $a$ and return a $s'$ and a $a'$. Next we perform the gradient descent update with the loss defined from the trajectory quadruple $(s, a, r, s^{'})$;
- Each trajectory generates a target:
  - $\large Q_{target}=r+\gamma \displaystyle\max_{a'} Q_{\theta}^-(s', a')$;
  - The discount factor models that later rewards are less valuable than immediate rewards;
  - The $Q_{\theta}$ is "frozen";
- Deep Q-Learning is a generalization of Q-Learning. We create a neural network (QNN) to predict the best next step'
  - $Q_t+1(s,a) \leftarrow Q_t(s,a)$
- We achieve data efficiency in deep Q-learning by creating a experience replay buffer of the quadruples to sum them;
  - $\langle S_t,A_t, R_{t+1}, S_{t+1} \rangle, \langle S_{t+1}, A_{t+1}, R_{t+2}, S_{t+2} \rangle, ...$;
- After computing the Q-Value, you choose the option with the higher probability value of attending the policy;
- We replace the future rewards by the consistency of the previous rewards;
  > Another way to frame it is that essentially a Q-table (every state/action pair) contains the total reward for taking any action in any state. However if there are lots of states and/or actions then that Q-table is going to be massive!! Possibly millions/billions of values for Q(s,a). Wouldn't it be nice if we had a way to approximate that function. (hey, neural nets!!). The Bellman equation simply gives us approximate "labels" which enable us to train a neural network to estimate our Q-table at every point.
  > Definitely a tough subject - I think the root of your question is answered in the proof of optimality of the Bellman equation (which requires some fun math). Essentially you can prove/apply the Bellman equation as the optimal state-action function for a given policy (all of this is built on the Markov Decision Processes so that's somewhat of a necessary prerequisite). If you're interested, this PDF does a pretty good job at going step by step from MDPs to the Bellman equation applied to state-action functions (Q) (sections 1.1 through 1.4).

## Policy based algorithms

- In this approach we try to predict policy directly. In the value approach, we calculate the value to try to find the best performance of the policy;
- The policy network is a regular neural network;
- The policy network gradient update:
  - $\Large\theta'=\theta+h\left(\frac{G_t}{\pi_{\theta}(s_t,a_t)} \right)\nabla_{\theta}\pi_{\theta}(s_t, a_t)$
  - Positive G means a gradient ascent update, which increases the probability of action $a$ taken in state $s$. A negative one is the opposite;
  - The denominator of the equation is the probability of the action given a state;
  - $h$ is the learning rate;
  - The future is hidden in the quantity of $G_t$. You need the return of each possible scenario. The $G$ is a weighted sum. After you have the final result, then we can update the values. Using the Alpha-Go example you would have to play the full game to know what path to follow;

## Actor-Critic

- It uses both techniques previously explained;
- We have two networks: One for the policy (Actor) and one for the Q-Value (Critic). The environment provides the state for both (Actor and Critic) and the reward only for the Critic. The critic send its result for the actor, which then sends the action for the environment;
  - $\Large\theta'=\theta+h\left( \frac{Q_w(s,a)}{\pi_{\theta}(s,a)} \right)\nabla_{\theta}\pi_{\theta}(s, a)$
  - For each experience trajectory $(s, a, r)$ we perform a gradient descent;

## Model based

- Learn an environment model from experience samples;
  - $S_{t-1}, A_{t-1} \rightarrow R_t, S_t$, ...
  - With this we can create a model and speedup the training;

## Contextual bandits

- Multi-Armed Bandits is a RL without the state. You choose which action to take next and you get a distribution of each action, determined by $\large R \sim P(r|a)$;
  - The goal is to find the action with the largest expectation;
  - Teh return is equal of the reward;
  - The Q-value is the reward expectation given the action;
- Always choose the action with maximal expected reward;
- For that we use the Epsilon-Greedy Policy
  - For that we keep track of the mean rewards per action;
- Regret as a metric:
  - $\LARGE L_t=\frac{1}{n}\displaystyle\sum^n_{t=1}(Q^*-r_t)$;
- An example of a multi-armed bandits agent is an A/B testing that rewards the user that doesn't know which action will return the highest reward;
- Contextual bandits condition the rewards to the state or context that the agent is operating. The agent has more data points to take its decision;
- Recommender systems takes the context (User features like age, search history) and the action (product features like product and price). Send them through hidden layers, concatenate them, sendo to dense layers and then calculate the Q-Value, which is the probability of click, defined by $Q_{\theta}(x,a)$;
- If we have a change in the context which is not a consequence of the previous actions, we should retrain it using the experiences that the system already have;
- Context Bandits setup is similar to multi-bandits, but we add a state to it;
- The gradient descent is the same. The target is now simpler, because there is only 1 state and 1 action per episode. So the term with the next state and the next action becomes zero, so we can remove it from the $Q_{target}$ formula;
  - $\LARGE Q_{target}=r+\gamma \xcancel{\displaystyle\max_{a'} Q_{\theta}^-}(s', a') \therefore Q_{target}=r+\gamma (s', a')$

## Notes

- [CartPole with Actor-Critic example](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic);
- [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf);
- [Stanford course on RL](https://www.youtube.com/watch?v=FgzM3zpZ55o);
- https://www.youtube.com/playlist?list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74
- https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0
