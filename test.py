import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_excel(r'Sales.xlsx', engine='openpyxl')

# Convert the DataFrame to PyTorch tensors
item1 = torch.tensor(df['prA'].values, dtype=torch.float)
item2 = torch.tensor(df['prB'].values, dtype=torch.float)
item3 = torch.tensor(df['prC'].values, dtype=torch.float)
item4 = torch.tensor(df['prD'].values, dtype=torch.float)
revenue = torch.tensor(df['revenue'].values, dtype=torch.float)



def model(item1, item2, item3, item4, revenue):
    # Prior distributions for the regression coefficients
    alpha = pyro.sample("alpha", dist.Normal(0, 10))
    beta1 = pyro.sample("beta1", dist.Uniform(7000, 15000))
    beta2 = pyro.sample("beta2", dist.Normal(10000, 7000))
    beta3 = pyro.sample("beta3", dist.Normal(10000, 7000))
    beta4 = pyro.sample("beta4", dist.Normal(10000, 7000))
    sigma = pyro.sample("sigma", dist.Normal(0, 1))

    # Expected revenue
    mu = alpha + beta1 * item1 + beta2 * item2 + beta3 * item3 + beta4 * item4

    # Likelihood
    with pyro.plate("data", len(revenue)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=revenue)



# Use No-U-Turn Sampler (NUTS) for MCMC
nuts_kernel = NUTS(model)

# Run MCMC
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(item1, item2, item3, item4, revenue)

mcmc.summary()



# Get the posterior samples
samples = mcmc.get_samples()

# Plot the samples for each parameter
fig, axs = plt.subplots(6, figsize=(10, 20))

for i, (name, values) in enumerate(samples.items()):
    sns.histplot(values, ax=axs[i], kde=True)
    axs[i].set_title(name)

plt.tight_layout()
plt.show()


import arviz as az

# Convert Pyro's MCMC output to an ArviZ data structure
posterior_samples = mcmc.get_samples()
idata = az.from_pyro(mcmc)

# Use ArviZ to plot the posterior
az.plot_posterior(idata)
plt.show()
