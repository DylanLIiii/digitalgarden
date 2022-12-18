---
{"dg-publish":true,"permalink":"/personal-page/variational-autoencoder/"}
---

Variational autoencoders (VAEs) are a type of generative model that can be used to learn the distribution of a dataset and generate new samples from that distribution. They are composed of two parts: an encoder, which maps an input data point x to a latent representation z, and a decoder, which maps a latent representation z back to a reconstructed data point x'. The encoder and decoder are trained together in an unsupervised manner to minimize the reconstruction error between x and x', while also forcing the latent representation z to follow a specific distribution (such as a unit Gaussian).

VAEs consist of an encoder network that maps the input data to a distribution over the latent code, and a decoder network that maps the latent code to a distribution over the reconstructed data. The encoder network can be represented mathematically as:

$q_\phi(z|x) = \mathcal{N}(z|\mu_\phi(x), \sigma_\phi(x)^2I)$

Where:

-   $q_\phi(\cdot|x)$ is the encoder distribution over the latent code, parameterized by $\phi$
-   $\mu_\phi(x)$ and $\sigma_\phi(x)$ are the mean and standard deviation of the encoder distribution, respectively, which are functions of the input data x and the parameters $\phi$
-   $\mathcal{N}(\cdot|\mu, \sigma^2I)$ is the multivariate normal distribution with mean $\mu$ and diagonal covariance $\sigma^2I$

The decoder network can be represented mathematically as:

$p_\theta(x|z) = \mathcal{N}(x|\mu_\theta(z), \sigma_\theta(z)^2I)$

Where:

-   $p_\theta(\cdot|z)$ is the decoder distribution over the reconstructed data, parameterized by $\theta$
-   $\mu_\theta(z)$ and $\sigma_\theta(z)$ are the mean and standard deviation of the decoder distribution, respectively, which are functions of the latent code z and the parameters $\theta$
-   $\mathcal{N}(\cdot|\mu, \sigma^2I)$ is the multivariate normal distribution with mean $\mu$ and diagonal covariance $\sigma^2I$

VAEs are trained by minimizing the KL divergence between the encoder distribution $q_\phi(\cdot|x)$ and the prior distribution $p(z)$ over the latent code, which is typically assumed to be a standard normal distribution, as well as the reconstruction loss between the input data x and the reconstructed data $\hat{x}$:

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}\left[\text{KL}\left(q_\phi(z|x)||p(z)\right)\right] + \mathbb{E}_{q_\phi(z|x)}\left[\text{ReconstructionLoss}(x, \hat{x})\right]$$

Where:

-   $\mathbb{E}_{q_\phi(z|x)}$ denotes the expectation over the encoder distribution $q_\phi(z|x)$
-   $\text{KL}(q_\phi(z|x)||p(z))$ is the KL divergence between the encoder distribution and the prior distribution $p(z)$ over the latent code. For a standard normal prior, this term is given by:

$$\text{KL}(q_\phi(z|x)||p(z)) = \frac{1}{2}\sum_{i=1}^d \left[1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2\right]$$

Where $\mu_i$ and $\sigma_i^2$ are the mean and variance of the encoder distribution $q_\phi(z|x)$, respectively, and $d$ is the dimensionality of the latent code.

-   $\text{ReconstructionLoss}(x, \hat{x})$ is the reconstruction loss between the input data $x$ and the reconstructed data $\hat{x}$, which is typically calculated using a chosen loss function such as the binary cross-entropy loss for binary data or the mean squared error loss for continuous data.

By minimizing the VAE loss function, the encoder and decoder networks are updated to learn a latent representation of the data that captures its underlying structure and can be used to generate new samples that are similar to the data.

1.  Encoding: The encoder takes an input data point x and maps it to a latent representation z through the use of two neural networks, a "mean network" and a "variance network". The mean network maps x to the mean vector μ of the latent distribution, and the variance network maps x to the diagonal covariance matrix Σ of the latent distribution. The latent representation z is then sampled from this distribution using the reparameterization trick:

z = μ + Σ^(1/2) * ε

where ε is a random noise vector sampled from a unit Gaussian distribution.

2.  Reconstruction: The decoder takes the latent representation z and maps it back to a reconstructed data point x' through the use of a neural network. The reconstruction loss is typically calculated as the negative log likelihood of the reconstructed data point x' given the input data point x, using a chosen reconstruction function (such as the binary cross-entropy loss for binary data or the mean squared error loss for continuous data).
    
3.  Regularization: In addition to the reconstruction loss, VAEs also include a regularization term in the loss function to encourage the latent representation z to follow the desired distribution. This is done by minimizing the KL divergence between the latent distribution and the desired distribution. For example, if the desired distribution is a unit Gaussian, the KL divergence loss would be calculated as:
    

KL(q(z|x) || p(z)) = -∑_i [1 + log(Σ_i) - μ_i^2 - Σ_i]

where q(z|x) is the latent distribution and p(z) is the desired distribution.

Overall, the VAE loss function can be written as:

Loss = Reconstruction loss + Regularization loss