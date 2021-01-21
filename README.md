# Reference

[**Categorical Reparameterization with Gumbel-Softmax**](https://arxiv.org/abs/1611.01144)

[**The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables**](https://arxiv.org/abs/1611.00712)

[Bayesian Deep Learning with torch distributions](https://github.com/kampta/pytorch-distributions)

[Concrete VAE](https://github.com/daandouwe/concrete-vae)

# Implementation

Jang's approach : prior distribution as categorical distribution

```
python main.py --sampling=TDModel --kld=eric
```

Maddison's approach : prior distribution as Concrete Distribution

```
python main.py --sampling=TDModel --kld=madisson
```

Maddison's treatment : prior distribution as ExpConcrete Distribution

```
python main.py --sampling=TDModel --kld=madisson
```

# Results

Maddison's way is sensitive and hard to optimize, but Jang's trick is easy to optimize.

Furthermore, Maddison's treatment for numerical issue has no effect in discrete-VAE.