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

Let's check this with an example. We draw $$n$$ samples from a uniform distribution with $$\mathcal{N}(0,1)$$ and plot the frequencies of the appearances in a histogram:
```Python
# import some packages
import numpy as np
import matplotlib.pyplot as plt

# Generate n samples from uniform distribution
n = 1000
np.random.seed(2) # set seed for pseudo random generator
x = np.random.randn(n)

# plot uniform samples
plt.figure(1)
binning = plt.hist(x,bins=100)
plt.xlabel('x')
plt.ylabel('frequency')
plt.title('Histogram of normal distribution')
```

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/normal_distribution.png?raw=true)

Now let's look at the CDF. Since we work with $$n$$ samples drawn at random from the uniform distribution $$\mathcal{N}(0,1)$$ we can sum up the number of appearances of the samples per bin and divide by the total number of samples $$n$$ to compute the CDF. 

```Python
# compute CDF
y = np.append(0,np.cumsum(binning[0])/n)
# plot CDF
plt.figure(2)
plt.plot(binning[1], y)
plt.xlabel('x')
plt.ylabel('F_X(x)')
plt.title('CDF')
```
![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/CDF.png?raw=true)


Let's check the inverse transform sampling with an exponential distribution. The CDF is given by 

$$
F(x|\lambda) = \left\{\begin{array}{ll}
1-\exp^{-\lambda x} & x\geq 0,\\
0 &\ x < 0
\end{array}\right. 
$$



## 2. Rejection Sampling



