# Sampling Methods

Today I want to present different methods for sampling numbers at random from any probability distribution only using a uniform distribution with the `numpy.random.rand` function. But let's start with the basics.

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

- Uniform distribution $$\mathcal(U)(a,b)$$:
  - PDF: $$ f(x) = \frac{1}{b-a}$$ for $$x \in [a,b]$$ else $$f(x) = 0$$ 
  - CDF: $$ F(x) = 0$$ for $$ x < a$$, $$F(x) = \frac{x-a}{b-a}$$ for $$x \in [a,b]$$, $$F(x) = 1$$ for $$x>b$$
- Exponential distribution: 
  - PDF: $$ f(x|\lambda) = \lambda \exp^{-\lambda x}$$ for $$x\geq 0$$, else $$F(x\lambda) = 0$$ 
  - CDF: $$ F(x|\lambda) = 1-\exp^{-\lambda x}$$ for $$x\geq 0$$, else $$F(x\lambda) = 0$$ 
- Normal distribution $$\mathcal{N}(\mu,\sigma)$$: 
  - PDF: $$ f(x|\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp^{-\frac{1}{2} (\frac{x-\mu}{\sigma})^2 }$$ 
  - CDF: $$ \Phi(x) = \frac{1}{\sigma\sqrt{2\pi}} \int_{-\infty}^{x} \exp^{-t^2/2} dt $$

  

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

### Example 1 - Exponential distribution:
Let's check the inverse transform sampling with an exponential distribution. The CDF is given by $$ F(x|\lambda) = 1-\exp^{-\lambda x}$$ and hence the inverse by $$F^{-1}(x|\lambda) = -\frac{log(1-u)}{\lambda} $$ for $$x\geq 0$$. By taking samples from $$\mathcal{U}(0,1)$$ we now can generate samples from the exponential distribution. 
```python
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
plt.plot(np.linspace(0,1,1000),n/bins*np.ones(1000), color="orange") # what we expect, i.e. the PDF of the uniform distribution
plt.xlabel('x')
plt.ylabel('frequency')
plt.title('Histogram of uniform distribution')


# apply inverse transform sampling 
samples_exponential = -1/lmbd*np.log(1-samples_uniform)

#check exponential distribution
plt.figure(1)
plt.hist(samples_exponential, density=True ,bins=bins)
plt.plot(x,lmbd*np.exp(-lmbd*x), color="orange") # what we expect, i.e. the PDF of the exponential distribution
plt.xlabel('x')
plt.ylabel('frequency')
plt.title('Histogram of exponential distribution')
```
With a histogram plot, we count the number of appearances of samples within one bin, where the bins have an equal length. As shown in the figure below we see the expected number of samples per bin in orange and the actual number of samples per bin fluctuating around this line. For the chosen parameters ($$n=5000$$ samples and $$bins=50$$) we expect $$100$$ samples per bin.

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/uniform_distribution.png?raw=true)

Now we plot the inverse transformed samples in a histogram and compare the distribution with the expected exponential probability density function shown in orange in the figure below. Looks quite the same, congrats you just generated exponentially distributed samples from a standard uniform distribution $$\mathcal{U}(0,1)$$.

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/exponential_distribution.png?raw=true)

### Example 2 - Normal distribution:
Ok, that was easy, but how would you deal with a normal distribution? The normal distribution has no closed-form expression for the inverse CDF. But we can use the *percent point function* from the `scipy.stats` package to generate normal distributed samples. 

```pyhton
from scipy.stats import norm
samples_normal = norm.ppf(samples_uniform) # apply inverse transform sampling

x = np.linspace(-5,5,1000) # to plot the expected curves
plt.figure(4)
plt.hist(samples_normal,density=True, bins=bins)
plt.plot(x,1/(np.sqrt(2*np.pi))*np.exp(-np.square(x)/2), color="orange") # what we expect, i.e. the PDF of the normal distribution
plt.xlabel('x')
plt.ylabel('frequency')
plt.title('Histogram of normal distribution')
```
By plotting the histogram of the generated samples as shown below we see that the generated samples follow a standard normal distribution $$\mathcal{N}(0,1)$$ (in orange). 

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/normal_distribution.png?raw=true)

## 2. Rejection Sampling
Instead of dealing with this percent point function from the last example, we can also use another method called rejection sampling. This method uses a uniform sampling of the two-dimensional space and keeps the samples below the PDF. 

### Example 1 - Normal distribution:

First, we sample the x-dimension again with $$\mathcal{U}(0,1)$$ and scale the samples to reach values from $$(a,b)$$. Second, we sample the y-dimension also with $$\mathcal{U}(0,1)$$ and now scale it to a range of $$(0,y_{max})$$.\
By applying the scaling we ensure samples cover the two-dimensional space sufficient for a standard uniform distribution. Note, that the 2-dimensional space is not sampled homogeneously by the scaling, but this shouldn't be a problem since we're just accepting the right samples. 

```python
import numpy as np
import matplotlib.pyplot as plt

# set parameters
n = 1000
a = -5
b = 5
y_max = 0.5

# 2D sampling of the space
x_samples = (b-a)*np.random.rand(n) + a
y_samples = y_max*np.random.rand(n)

# check for acceptance
accept = np.zeros(n,dtype=bool)
for i in range(n):
    if y_samples[i] <= 1/(np.sqrt(2*np.pi))*np.exp(-np.square(x_samples[i])/2):
        accept[i] = True

# plot rejection sampling results
x = np.linspace(a,b,1000) # to plot the expected curves
plt.figure(5)
plt.plot(x,1/(np.sqrt(2*np.pi))*np.exp(-np.square(x)/2), color="orange") # what we expect, i.e. the PDF
plt.plot(x_samples[accept],y_samples[accept],'.',color='green') # the samples we keep
plt.plot(x_samples[np.invert(accept)],y_samples[np.invert(accept)],'x', color='red') # the samples we discard
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rejection sampling of normal distribution')
```

In the below figure, the accepted samples are shown in green, and the rejected ones are in red. 

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/rejection_sampling.png?raw=true)

Taking the accepted samples and plotting the historam to check for the normal distribution:

```python
samples_normal = x_samples[accept]
plt.figure(6)
plt.hist(samples_normal,density=True, bins=bins)
plt.plot(x,1/(np.sqrt(2*np.pi))*np.exp(-np.square(x)/2), color="orange") # what we expect
plt.xlabel('x')
plt.ylabel('frequency')
plt.title('Histogram of normal distribution by rejection sampling')
```
Looks ok, but not as close as the other histogram plots. This is because we just got $$209$$ accepted samples.

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/normal_distribution_rejection_sampling.png?raw=true)

The number of samples decreasing to some unforeseen amount is one major drawback of the rejection sampling method. The number will even further decrease if the initial scaling is not set properly. Let's try to increase the initial number of two-dimensional samples to $$n=5000$$ and plot the histogram. Now it looks quite nice.

![alt text](https://github.com/ludwigwaibel/ludwigwaibel.github.io/blob/main/_img/sampling/normal_distribution_rejection_sampling_n_5000.png?raw=true)


