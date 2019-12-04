## Notes In Deep Reinforcement Learning 
---

### Concepts in Reinforcemnet Learning
1. The main goal of Reinforcement Learning is to maximum the `Total Reward`.
2. `Total Reward` is the sum of all reward in `One Eposide`, so the model doesn't know which steps in this eposide are good and which are bad.
3. Only few actions can get the positive reward (ex: fire and killing the enemy in Space War game gets positive reward but moving gets no reward), so how to let the model find these right actions is very important.


### Difficulties in RL
1. Reward Delay<br>
   * Only "Fire" can obtain rewards, but moving before fire is also important (moving has no reward), how to let the model learn to move properly?
   * In chess game, it may be better to sacrifice immediate reward to gain more long-term reward.
2. Agent's actions may affect the environment
   * How to explore the world (`observation`) as more as possible.
   * How to explore the `action-combination` as more as possible.

### A3C Method - Asynchronous Advantage Actor-Critic
The A3C method is the most popular model which combines policy-based method and value-based method, the structure is shown as below. To learn A3C model, we need to know the concepts of policy-based and value-based.

<div align=center><img src="assets/A3C.png"></div>

#### Policy-based Approch - Learn an Actor
This approch try to learn a policy(also called actor). It accepts the observation as input, and output an action. The policy(actor) can be any model. If you use an Neural Network to as your actor, then you are doing Deep Reinforcemnet Learning.

$$Input(Observation) \rightarrow Actor/Policy \rightarrow Output(Action)$$

There are three steps to build DRL:<br>
**1. Decide Function of Actor Model (NN? Q-Table?...)**<br>
Here we use the NN as our Actor, so:
* The Input of this NN is the observation of machine represented as Vector or Matrix. (Ex: Image Pixels to Matrix)
* The Output of this NN is Action `Probability`. The most important point is that we shouldn't always choose the action which has the highest probability, it should be a stochastic decisions according to the probability distrubution.

<div align=center><img src="assets/NN_ACtor.png"></div>

**2. Decide Goodness of this Function**
**3. Choose the best function**