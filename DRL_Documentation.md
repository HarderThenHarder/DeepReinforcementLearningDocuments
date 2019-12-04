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
* The Advantage of NN to Q-table is: we can't enumerate all observations (such as we can't list all pixels' combinations of a game) in some complex scenes, then we can use Neural Network to promise that we can always obtain an output even if this observation didn't apprear in the previous train set.

<div align=center><img src="assets/NN_ACtor.png"></div>

**2. Decide Goodness of this Function**<br>
Since we use the Neural Network as our function model, we need to decide what is the goodness of this model (a standard to judge the performance of current model). We use $\overline{R(\theta)}$ to express this standard, which $\theta$ is the parameters of current model.

* Given an actor $\pi_\theta(t)$ with Network-Parameters $\theta$, $t$ is the observation (input).
* Use the actor $\pi_\theta(t)$ to play the video game until this game finished.
* Sum all rewards in this eposide and marked as $R(\theta) \rightarrow R(\theta) = \sum_{t=1}^Tr_t$.
`Note`: $R(\theta)$ is a variable, cause even if we use the same actor $\pi_\theta(t)$ to play the same game many times, we can get the different $R(\theta)$ (*random mechanism in game and action chosen*). So we want to maximum the $\overline{R(\theta)}$ which expresses the expect of $R(\theta)$.
* Use the $\overline{R(\theta)}$ to expresses the goodness of $\pi_\theta(t)$.

*How to Calculate the $R(\theta)$?*

* An eposide is considered as a tractory $\tau$
  * $\tau$ = {$s_1, a_1, r_1, s_2, a_2, r_2, ..., s_T, a_T, r_T$} $\rightarrow$ *all the history in this eposide*
  * $R(\tau) = \sum_{t=1}^Tr_t$

* Different $\tau$ has different probability to appear, the probability of $\tau$ is depending on the parameter $\theta$ of actor $\pi_\theta(t)$. So we define the probability of $\tau$ as $P(\tau|\theta)$.
$$\overline{R(\theta)} = \sum_\tau{P(\tau|\theta)R(\theta)}$$

* We use actor $\pi_\theta(t)$ to play N times game, obtain the list {$\tau^1, \tau^2, ..., \tau^N$}. Each $\tau^n$ has a reward $R(\tau^n)$, the mean of these $R(\tau^n)$ approximatly equals to the expect $\overline{R(\theta)}$.
$$\overline{R(\theta)} \approx \frac{1}{N}\sum_{n=1}^NR(\tau^n)$$ 

**3. Choose the best function**<br>
Now we need to know how to calculate the $\theta$, here we use the `Gradient Ascend` method.

* problem statements:
$$\theta^* = argmax_\theta\overline{R(\theta)} \rightarrow \overline{R(\theta)} = \sum_{\tau}P(\tau|\theta)R(\theta)$$
* gradient ascent:
  * Start with $\theta^0$.
  * $\theta^1 = \theta^0 + \eta\bigtriangledown{\overline{R(\theta^0)}}$
  * $\theta^2 = \theta^1 + \eta\bigtriangledown{\overline{R(\theta^1)}}$
  *  ...
* The $\theta$ includes the parameters in the current Neural Network, $\theta = $ {$w_1, b_1, w_2, b_2, w_3, b_3, ...$}, which the $\bigtriangledown R(\theta) = \left[ \begin{matrix} \frac{\partial{R(\theta)}}{\partial{w_1}} \\ \frac{\partial{R(\theta)}}{\partial{w_2}} \\ ... \\ \frac{\partial{R(\theta)}}{\partial{b_1}} \\ \frac{\partial{R(\theta)}}{\partial{b_2}} \\ ... \end{matrix} \right]$.