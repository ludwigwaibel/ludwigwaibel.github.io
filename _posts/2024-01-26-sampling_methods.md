# Sampling Methods

Today I want to present different methods for sampling numbers at random from any probability distribution only using a uniform distribution over $$[0, 1)$$ with the `numpy.random.rand` function. But let's start with the basics.

## 0. Definitions
First I'll give you some definitions used in this post real quick.

- Probability: \
Let $$S$$ be a sample space, $$\mathcal{B}$$ a sigma algebra, and $$P$$ a probability function. Then $$P$$ satisfies:

$$ 
\begin{align}
& P(A) \leq 1 \quad \forall A \in \mathcal{B}\\
& P(S) = 1\\
& A_1, A_2,... \in \mathcal{B} \text{ disjoint } \Rightarrow P\left(\bigcup_{i = 1}^\infty A_i \right) = \sum_{i=1}	^\infty P(A_i)
\end{align}
$$
- Cumulative distribution function (CDF):
  
$$
F_X(x) = P_X(X \leq x) \quad \forall x
$$

- Probability density function (PDF):

$$
F_X(x) = \int_{-\infty}^x f_X(t) dt
$$
  

## 1. Inverse Transform Sampling
The inverse transform sampling is a basic method for pseudo-random number sampling, i.e., for generating sample numbers at random from any probability distribution given its cumulative distribution function.
The method follows directly from the Probability Integral Transform: 

> Let $$X$$ be a random variable with continuous distribution and CDF $$F_X(x)$$ then the random variable $$Y= F_X(x)$$ has a standard uniform distribution, i.e. $$Y \sim \mathcal{U}(0,1)$$.

Proof:
  
$$
\begin{align}
P(Y \leq y) & = P(F_X(X) \leq y) \\
& = P(F_X^{-1} F_X(X) \leq F_X^{-1}(y))\\
& = P(X \leq  F_X^{-1}(y))\\
& = F_X(F_X^{-1}(y))\\
&= y
\end{align}
$$

### Example:
Let's check the inverse transform sampling with an exponential distribution. The CDF is given by $$ F(x|\lambda) = 1-\exp^{-\lambda x}$$ and the inverse by $$F^{-1}(x|\lambda) = -\frac{log(1-u)}{\lambda} $$ for $$x\geq 0$$. By taking samples from $$\mathcal{U}(0,1)$$ we now can generate samples from the exponential distribution. 
```Python
# import libraries
import numpy as np
import matplotlib.pyplot as plt

# set parameters
lmbd = 0.75 # lambda for the exponential distribution
n = 5000 # number of samples to draw
bins = 50 # number of bins for the histogram
x = np.linspace(0,20,1000) # to plot the expected curves
np.random.seed(0) # set seed of pseudo random generator

# generate uniform samples
samples_uniform = np.random.rand(n)

#check uniform distribution
plt.figure(0)
plt.hist(samples_uniform, bins=bins)
plt.plot(np.linspace(0,1,1000),n/bins*np.ones(1000), color="orange") # what we expect
plt.xlabel('x')
plt.ylabel('frequency')
plt.title('Histogram of uniform distribution')


# apply inverse transform sampling 
samples_exponential = -1/lmbd*np.log(1-samples_uniform)

#check exponential distribution
plt.figure(1)
plt.hist(samples_exponential, density=True ,bins=bins)
plt.plot(x,lmbd*np.exp(-lmbd*x), color="orange") # what we expect
plt.xlabel('x')
plt.ylabel('frequency')
plt.title('Histogram of exponential distribution')
```
For the chosen parameters ($$n=5000$$ samples and $$bins=50$$) we expect $$100$$ samples per bin. As shown in the figure below we see the expected number of samples per bin in orange and the actual number of samples per bin fluctuating around this line. 

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/uniform_distribution.png?raw=true)



## 2. Rejection Sampling



