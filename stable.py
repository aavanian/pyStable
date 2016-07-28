"""
Stable Distributions
-------------------------
"""
# Author: F.J. Martinez-Murcia <fjesusmartinez@ugr.es>. Based on Levy Stable code by Paul F. Harrison

import numpy as np
import scipy.special as sp
from sklearn.base import BaseEstimator


_lower = np.array([-np.pi / 2 * 0.999, 0.5, -1.0])
_upper = np.array([np.pi / 2 * 0.999, 2.0, 1.0])
    
def _reflect(x, lower, upper):
    while 1:
        if x < lower:
            x = lower - (x - lower)
        elif x > upper:
            x = upper - (x - upper)
        else:
            return x
            
par_names = ['alpha', 'beta', 'mu', 'sigma']
default = [1.5, 0.0, 0.0, 1.0]
default = {par_names[i]: default[i] for i in range(4)}
f_bounds = {
    'alpha': lambda x: _reflect(x, _lower[1], _upper[1]),
    'beta': lambda x: _reflect(x, _lower[2], _upper[2]),
    'mu': lambda x: x,
    'sigma': lambda x: x
}

class StableDist(BaseEstimator):
    
    """Stable Distribution fitting. Currently it works only for 
    $0.5<\alpha<2.0$. 

    Read more in the :ref:`User Guide <stable_distribution>`.

    Parameters
    ----------
    alpha: 
        The $\alpha$ value of the Alpha-Stable distribution. $0.5<\alpha<2.0$
    beta:
        The $\beta$ parameter of the alpha-stable distribution. $-1<\beta<1$
    mu: 
        The $\mu$ parameter (also known as $\gamma$ or location) of the alpha-stable dist. 
    sigma: 
        The $\sigma$ parameter (also known as $\delta$ or scale) of the alpha stable dist.
    """
    def __init__(self, alpha=2., beta=0., mu=0., sigma=1.):
        if alpha>=0.5 and alpha<=2.0:
            self.alpha = alpha
        else:
            raise Exception('Alpha must be between 0.5 and 2.0')
        if beta>=-1. and beta<=1.0:
            self.beta = beta
        else:
            raise Exception('Beta must be between -1.0 and 1.0')
        self.mu = mu
        if sigma>0:
            self.sigma = sigma
        else:
            raise Exception('Sigma must be higher than 0')
        self.phi = self.beta*np.tan(np.pi*self.alpha/2)
        
    def _update_params(self, alpha, beta, mu, sigma):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.phi = self.beta*np.tan(np.pi*self.alpha/2)
        
    def _calculate_phi(self, t, x):
        """
        Calculates the characteristic function
        """
        if self.alpha==1:
            Phi = -2*np.log(t)
        else:
            Phi = np.tan(np.pi*self.alpha/2)
        phi = phi = np.exp(-1j*t*x + 1j*t*self.mu - np.power(np.abs(self.sigma*t),self.alpha)*(1-1j*self.beta*Phi*np.sign(t)))
        return phi
        
#    def _calculate_levy(self, x, cdf=False):
#        from scipy import integrate
#        if cdf:
#            jarl = integrate.nquad(self._calculate_phi, [[1, np.inf],[0, np.inf]],args=(self.alpha, self.beta, self.mu, self.sigma, x))
#            jarl = integrate.dblquad(self._calculate_phi(self, t, x), )
#        else:
#            jarl = integrate.quad(self._calculate_phi, 0.0, np.inf, args=(self.alpha, self.beta, self.mu, self.sigma, x))[0]/(2*np.pi)
#        return jarl
        
    def _calculate_levy(self, x, alpha, beta, cdf=False):
        """ 
        Creates a lookup table of the stable distribution, 
        via numerical integration. It uses the "0" parameterization as in 
        http://academic2.americanp.edu/~jpnolan/stable/stable.html
        Note: fails for alpha=1.0
               (so make sure alpha=1.0 isn't exactly on the interpolation grid)
        """
        from scipy import integrate
        
        C = beta*np.tan(np.pi*alpha/2)
        
        def func_cos(u):
            ua = u ** alpha
            if ua > 700.0: return 0.0
            return np.exp(-ua) * np.cos(C * ua - C * u)
    
        def func_sin(u):
            ua = u ** alpha
            if ua > 700.0: return 0.0
            return np.exp(-ua) * np.sin(C * ua - C * u)
    
        if cdf:
            # Cumulative density function
            return (integrate.quad(lambda u: u and func_cos(u) / u or 0.0, 0.0, np.inf, weight="sin", wvar=x,
                                   limlst=1000)[0]
                    + integrate.quad(lambda u: u and func_sin(u) / u or 0.0, 0.0, np.inf, weight="cos", wvar=x,
                                     limlst=1000)[0]
                    ) / np.pi + 0.5
        else:
            # Probability density function
            return (integrate.quad(func_cos, 0.0, np.inf, weight="cos", wvar=x, limlst=1000)[0]
                    - integrate.quad(func_sin, 0.0, np.inf, weight="sin", wvar=x, limlst=1000)[0]
                    ) / np.pi



    def _levy_tan(self, x, alpha, beta, cdf=False):
        """ Calculate the values stored in the lookup table. 
            The tan mapping allows the table to cover the range from -INF to INF. """
        x = np.tan(x)
        return self._calculate_levy(x, alpha, beta, cdf)
    
    
    def _interpolate(self, points, grid, lower, upper):
        """ Perform multi-dimensional Catmull-Rom cubic interpolationp. """
        point_shape = np.shape(points)[:-1]
        points = np.reshape(points, (np.multiply.reduce(point_shape), np.shape(points)[-1]))
    
        grid_shape = np.array(np.shape(grid))
        dims = len(grid_shape)
    
        points = (points - lower) * ((grid_shape - 1) / (upper - lower))
    
        floors = np.floor(points).astype('int')
    
        offsets = points - floors
        offsets2 = offsets * offsets
        offsets3 = offsets2 * offsets
        weighters = [
            -0.5 * offsets3 + offsets2 - 0.5 * offsets,
            1.5 * offsets3 - 2.5 * offsets2 + 1.0,
            -1.5 * offsets3 + 2 * offsets2 + 0.5 * offsets,
            0.5 * offsets3 - 0.5 * offsets2,
        ]
    
        ravel_grid = np.ravel(grid)
    
        result = np.zeros(np.shape(points)[:-1], 'float64')
        for i in range(1 << (dims * 2)):
            weights = np.ones(np.shape(points)[:-1], 'float64')
            ravel_offset = 0
            for j in range(dims):
                n = (i >> (j * 2)) % 4
                ravel_offset = ravel_offset * grid_shape[j] + \
                               np.maximum(0, np.minimum(grid_shape[j] - 1, floors[:, j] + (n - 1)))
                weights *= weighters[n][:, j]
    
            result += weights * np.take(ravel_grid, ravel_offset)
    
        return np.reshape(result, point_shape)
    
    
    def _approximate_pdf(self, x):
        return (1.0 + np.abs(self.beta)) * np.sin(np.pi * self.alpha / 2.0) * \
               sp.gamma(self.alpha) / np.pi * np.power(x, -self.alpha - 1.0) * self.alpha
    
    
    def _approximate_cdf(self, x):
        return 1.0 - (1.0 + np.abs(self.beta)) * np.sin(np.pi * self.alpha / 2.0) * \
                     sp.gamma(self.alpha) / np.pi * np.power(x, -self.alpha)
    
    
    def _make_data_file(self):
        """ Generates the lookup table, writes it to a numpy npz file. """
    
        size = (200, 50, 51)
        pdf = np.zeros(size, 'float64')
        cdf = np.zeros(size, 'float64')
        xs, alphas, betas = [np.linspace( _lower[i], _upper[i], size[i], endpoint=True) for i in range(len(size))]
    
        print("Generating levy_data.py ...")
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                print("Calculating alpha={}, beta={}".format(alpha, beta))
                for k, x in enumerate(xs):
                    pdf[k, i, j] = self._levy_tan(x, alpha, beta)
                    cdf[k, i, j] = self._levy_tan(x, alpha, beta, True)
    
        np.savez('levy_data.npz', pdf=pdf, cdf=cdf)
    
    
    def _int_levy(self, x, alpha, beta, cdf=False):
        """ Interpolate densities of the Levy stable distribution specified by alpha and beta.
    
            Specify cdf=True to obtain the *cumulative* density functionp.
    
            Note: may sometimes return slightly negative values, due to numerical inaccuracies.
        """
        levy_data = np.load('levy_data.npz')
    
        points = np.empty(np.shape(x) + (3,), 'float64')
        points[..., 0] = np.arctan(x)
        points[..., 1] = alpha
        points[..., 2] = beta
    
        if cdf:
            what = levy_data['cdf']
        else:
            what = levy_data['pdf']
        return self._interpolate(points, what,  _lower, _upper)
    
    
    def _get_closest_approx(self, alpha, beta):
        x0, x1, n = -50.0, 10000.0 - 50.0, 100000
        dx = (x1 - x0) / n
        x = np.linspace(x0, x1, num=n, endpoint=True)
        y = 1.0 - self._int_levy(x, alpha, beta, cdf=True)
        z = 1.0 - self._approximate_cdf(x, alpha, beta)
        mask = (10.0 < x) & (x < 500.0)
        return 10.0 + dx * np.argmin((np.log(z[mask]) - np.log(y[mask])) ** 2.0)
    
    
    def _make_approx_data_file(self):
   
        size = (50, 51)
        limits = np.zeros(size, 'float64')
        alphas, betas = [
            np.linspace( _lower[1], _upper[1], size[0], endpoint=True),
            np.linspace(0, _upper[2], size[1], endpoint=True)]
    
        print("Generating levy_approx_data.npz ...")
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                limits[i, j] = self._get_closest_approx(alpha, beta)
                print("Calculating alpha={}, beta={}, limit={}".format(alpha, beta, limits[i, j]))
    
    
        np.savez('levy_approx_data.npz', limits=limits)

    
    def levy(self, x, alpha=None, beta=None, mu=None, sigma=None, cdf=False, par=0):
        """
        Levy with the tail replaced by the analytical approximation.
        Also, mu, sigma are parameters that shift and rescale the distribution.

        Parameters
        ----------
        x:
            Data to be interpolated. 
        alpha: 
            The $\alpha$ value of the Alpha-Stable distribution. Higher than 0.5.
        beta:
            The $\beta$ parameter of the alpha-stable distribution.
        mu: 
            The $\mu$ parameter (also known as $\gamma$ or location) of the alpha-stable dist. 
        sigma: 
            The $\sigma$ parameter (also known as $\delta$ or scale) of the alpha stable dist.
        cdf: 
            Whether or not we want to obtain the pdf (False), or cdf (True)
        par: 
            Parametrization can be chosen according to Nolan, par={0,1}.
        """
        if alpha==None:
            alpha=self.alpha
        if beta==None:
            beta=self.beta
        if mu==None:
            mu=self.mu
        if sigma==None:
            sigma=self.sigma
    
        if par == 0:
            loc = mu
        elif par == 1:
            loc = mu+ beta * sigma* np.tan(np.pi * alpha / 2.0)  # Par 1 is changed
        
#        import os
#        os.chdir('pylevy')
        levy_data = np.load('levy_data.npz') #Loads datafiles. 
        levy_approx_data = np.load('levy_approx_data.npz')
    
        if cdf:
            what = levy_data['cdf']
            app = self._approximate_cdf
        else:
            what = levy_data['pdf']
            app = self._approximate_pdf
        limits = levy_approx_data['limits']
    
        xr = (x - loc) / sigma
        beta = -beta
        alpha_index = int((alpha - 0.5) * 49.0 / 1.5)
        beta_index = int(np.abs(beta) * 50.0)
        l = limits[alpha_index, beta_index]
        if beta <= 0.0:
            mask = (xr < l)
        elif beta > 0.0:
            mask = (xr > -l)
        z = xr[mask]
    
        points = np.empty(np.shape(z) + (3,), 'float64')
        points[..., 0] = np.arctan(z)
        points[..., 1] = alpha
        points[..., 2] = beta
    
        interpolated = self._interpolate(points, what,  _lower, _upper)
        approximated = app(xr[~mask])
    
        res = np.empty(np.shape(xr), 'float64')
        if cdf is False:
            interpolated = interpolated / sigma
            approximated = approximated / sigma
        res[mask] = interpolated
        res[~mask] = approximated
        return res
    
    
    def neglog_levy(self, x, alpha, beta, mu, sigma, par=0):
        """
        Interpolate negative log densities of the Levy stable distribution specified by alpha and beta.
        Small/negative densities are capped at 1e-100 to preserve sanity.
        """
        return -np.log(np.maximum(1e-100, self.levy(x, alpha, beta, mu, sigma, par=par)))
    
    
    def fit(self, x, par=0, disp=False, maxiter=1e6):
        """
        Fit parameters of Levy stable distribution given data x, using an 
        initial guess based on SLSQP and finetuning by Truncated Newton 
        algorithm. 

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
            
    
        Returns a tuple of ([alpha, beta, mu, sigma], negative log density)
        """
    
        from scipy import optimize
        
#        # Use ordering to set the initial mu and sigma parameters. 
#        x = np.sort(x)
#        
#        last = len(x)-1
#        mu = x[last/2]
#        sigma = (x[last-last/4] - x[last/4])/2.0
#        
#        # Maybe there are lots of zeros or something...
#        if sigma == 0:
#            sigma = (x[last] - x[0]) / 2.0
#            
#        parameters = [self.alpha, self.beta, mu, sigma]
        parameters = [self.alpha, self.beta, self.mu, self.sigma]
        
        def neglog_density(param):
            return np.sum(self.neglog_levy(x, param[0], param[1], param[2], param[3]))
        
        #Initial guess
        jarl = optimize.minimize(neglog_density, parameters, method='L-BFGS-B', bounds=((0.5, 2),(-1,1), (None,None),(1e-5,None)) ,options={'maxiter': maxiter, 'disp':disp})
        print(jarl.x)
        #Finetuning
        parameters = optimize.minimize(neglog_density, jarl.x, method='TNC', bounds=((0.5, 2),(-1,1), (None,None),(1e-5,None)) ,options={'maxiter': 100, 'disp':disp})

        
        self._update_params(parameters.x[0], parameters.x[1], parameters.x[2], parameters.x[3])
        
        return (parameters.x, parameters.fun)
    
    
    def sample(self, n_samples=1, random_state=None, par=0):
        """
        Generate random values sampled from an alpha-stable distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        random_state : RandomState or an int seed (0 by default)
            A random number generator instance.
            
        par : 
            Parametrization can be chosen according to Nolan, par={0,1}.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples.
        """
        if random_state is not None:
            np.random.seed(seed=random_state)
        if par == 0:
            mu0 = self.mu
        elif par == 1:
            mu0 = self.mu+ self.beta * self.sigma* np.tan(np.pi * self.alpha / 2.0)  # Par 1 is changed
    
        if self.alpha == 2:
            return np.random.randn(n_samples) * np.sqrt(2.0)
    
        # Fails for alpha exactly equal to 1.0
        # but works fine for alpha infinitesimally greater or less than 1.0  
        alpha = self.alpha
        radius = 1e-15  # <<< this number is *very* small
        if np.absolute(alpha - 1.0) < radius:
            # So doing this will make almost exactly no difference at all
            alpha = 1.0 + radius
    
        r1 = np.random.random(n_samples)
        r2 = np.random.random(n_samples)
        pi = np.pi
    
        a = 1.0 - alpha
        b = r1 - 0.5
        c = a * b * pi
        e = self.phi
        f = (-(np.cos(c) + e * np.sin(c)) / (np.log(r2) * np.cos(b * pi))) ** (a / alpha)
        g = np.tan(pi * b / 2.0)
        h = np.tan(c / 2.0)
        i = 1.0 - g ** 2.0
        j = f * (2.0 * (g - h) * (g * h + 1.0) - (h * i - 2.0 * g) * e * 2.0 * h)
        k = j / (i * (h ** 2.0 + 1.0)) + e * (f - 1.0)
    
        return mu0 + self.sigma* k
    
    
