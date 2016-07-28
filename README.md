# pyStable
Library based on pyLevy for working with Alpha-Stable distributions. Based on the library implemented by Paul Harrison, it features python 3.5 support, and a class interface similar to the one used in sklearn for estimating KDE. 

## Example of use:
1. Create a StableDist object: 
```python
import stable as stb
est = stb.StableDist(alpha=0.5, beta=-0.5, mu=1, sigma=0.1)
```

2. Use the object to estimate the distribution of a dataset: 
```python
from scipy.stats import norm
import numpy as np
x = np.random.randn(3000)
est.fit(x)
```
3. And plot the data
```python
ejex=np.arange(-5,5,0.01)
plt.plot(ejex,norm.pdf(ejex), label='Original')
plt.plot(ejex,est.levy(ejex), label='Estimated')
plt.legend()
```
