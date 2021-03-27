import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import logging
import dill

"""
 Copyright 2021 Technical University of Denmark
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

Author: Tommy Sonne Alstrøm <tsal@dtu.dk> 
(translated to Python and extended functionality by David Frich Hansen <dfha@dtu.dk>)
 $Date: 2021/03/04$

 generate simulated raman data

"""



class SERSGenerator:
    def __init__(self, mapsize, Nw, seed=None, c=None, gamma=None, eta=None, hotspot_loc=None):
        self.mapsize = mapsize
        self.Nw = Nw
        self.w = np.arange(Nw)

        self.seed = seed

        self.hotspot_loc = hotspot_loc

        if c is not None:
            self.K = len(c)

            self.c = c
        
        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = None
        
        if eta is not None:
            self.eta = eta
        else:
            self.eta = None
            
        if c is not None:
            self.c = c
        else:
            self.c = None


    def pseudo_voigt(self, w, c, gamma, eta):
        """Computes the (scaled) Pseudo-Voigt function over wavenumbers w with parameters c, gamma, eta efficiently.

        V = eta * Lorentzian + (1-eta) * Gaussian

        Args:
            w (np.array): Measured wavenumbers (sorted) (W x 1)
            c (np.array): Centers of Voigt curves (K x 1)
            gamma (np.array): Full-width-at-half-maximum (FWHM) of Voigt curves (K x 1)
            eta (np.array): Mixture coefficients of Lorentzian and Gaussian (0 <= eta <= 1) (K x 1)

        Returns:
            np.ndarray: Computed Voigt curves, (K x W)
        """
        K = len(c)
        assert len(gamma) == K and len(eta) == K
        W = len(w)

        xdata = np.tile(w, (K, 1))
        c_arr = np.tile(c, (W, 1)).T
        gamma_arr = np.tile(gamma, (W, 1)).T
        eta_arr = np.tile(eta, (W, 1)).T

        diff = xdata - c_arr
        kern = diff / gamma_arr
        diffsq = kern * kern

        # lorentzian
        L = 1 / (1 + diffsq)

        # gaussian
        G_kern = -np.log(2) * diffsq
        G = np.exp(G_kern)

        # vectorized voigts
        Vo = eta_arr * L + (1 - eta_arr) * G

        return Vo


    def generate(self, N_hotspots, sig, sbr, K=None, p_outlier=0,  background='default', plot=True, smooth=False, reset=False, plot_matrix=False,
                *bgargs, **bgkwargs):
        """Generates a SERS map. Also sets several attributes in the object with generated quantities for
        later retrieval.


        Args:
            N_hotspots: Number of hotspots on the plate (int > 0)
            K:          Number of peaks (int > 0)
            sig:        Measurement error (float > 0)
            sbr:        Signal-to-background ratio (float > 0)
            p_outlier:  Probability of a given spectrum being an outlier (ie outside hotspot) (0 <= float <= 1)
            background: String or callable.
                        String options:
                            'default' generates background as AR(1) process.
                            'none' generates no background.
                        Callable should have signature
                        background(self.w, *bgargs, **bgkwargs) -> np.ndarray of size self.Nw x 1.
                        Note that in this case no further computations are done on the background signal - only basic
                        input checks are done.
                        In particular, this means that all inherent stochastic elements of such a signal *must* be
                        handled by the callable itself.

            plot:       Plot generated qunatities on generation (boolean)
            smooth:     Should Savitzky-Golay background smoothing be applied? (boolean)
                        Only available if background == 'default'
            reset:      Specifies whether the generation should be reset, such that it runs with variables set to None
            *bgargs:    Any extra arguments to background if callable(background) == True.
            **bgkwargs: Any extra keyword-only arguments to background if callable(background) == True

        Returns:
            X:          Generated SERS map. (np.ndarray of size (self.nw, np.prod(mapsize))
        """
        if reset:
            self.c = None
            self.eta = None
            self.gamma = None
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.c is None and K is None:
            raise ValueError('K needs to be specified if c, gamma and eta are not prespecified')

        if self.c is not None and K is None:
            logging.warning('c, gamma and eta are prespecified, so argument for K is ignored')
            K = len(self.c)

        N = np.prod(self.mapsize)
        DD = np.zeros((N, self.Nw))
        LL = np.zeros((N, self.Nw))

        # measurement noise
        self.sigma = sig

        ### Generate outliers
        if p_outlier > 0:
            self.outliers = True
            NL = np.random.binomial(N, p_outlier)
            eta = np.random.randn(NL, self.Nw)
            L = np.zeros_like(eta)
            L[:,0] = np.random.rand(NL)
            # c could be changed here
            c = 0
            phi = 1
            for w in range(1, self.Nw):
                L[:,w] = c + phi*L[:, w-1] + eta[:,w]

            L -= repmat(np.min(L, axis=1)[:,np.newaxis], 1, self.Nw)
            L /= repmat(np.max(L, axis=1)[:, np.newaxis], 1, self.Nw)

            l = np.random.exponential(1e-2, size=(NL,1))
            L *= repmat(l, 1, self.Nw)

            if plot:
                plt.figure()
                plt.title('Outliers')
                plt.plot(L.T)
                plt.show()

            inx = np.random.choice(N, NL)
            LL[inx, :] = L
            self.N_outliers = NL

        ### Generate background


        if background == 'default':
            # this can be changed based on application
            eta = np.random.randn(self.Nw, 1)
            B = np.zeros_like(eta)
            B[0] = np.random.rand()
            #c = 0.2
            c = 0.05
            phi = 0.995
            for w in range(1, self.Nw):
                B[w] = c + phi*B[w-1] + eta[w]
            if smooth:
                from scipy.signal import savgol_filter
                B = savgol_filter(B.ravel(), window_length=149, polyorder=2)


            if plot:
                plt.figure()
                plt.title('Background spectrum')
                plt.plot(range(self.Nw), B)
                plt.show()

            if len(B) > 1:
                B -= np.min(B)
                B /= np.max(B)
                B += np.random.rand()



        elif background == 'zero':
            if smooth:
                logging.warning('Smoothing available only for default background. Ignoring')
            self.B = np.zeros((1, self.Nw))

        elif callable(background):
            if smooth:
                logging.warning('Smoothing available only for default background. Ignoring')

            B = background(self.w, *bgargs, **bgkwargs)
            assert B.ndim == 1
            assert len(B) == self.Nw
            assert (B>=0).all()

            if plot:
                plt.figure()
                plt.title('Background spectrum')
                plt.plot(B)
                plt.show()
        else:
            raise ValueError("Illegal input for 'background'. Should be 'default', 'none' or a callable")

        self.B = B
        B = np.reshape(B, -1)
        B = repmat(B, N, 1)
        b = np.random.beta(100, 100, size=(N, 1))
        self.b = b

        b = repmat(b, 1, self.Nw)
        BB = b * B

        if plot:
            plt.matshow(B)
            plt.title('Background map')
            plt.colorbar()
            plt.show()

        ### Generate hotspots (signal)
        if N_hotspots > 0:
            if self.hotspot_loc is None:
                mu = repmat(self.mapsize, N_hotspots, 1) * np.random.rand(N_hotspots, 2)
                mu = np.rint(mu)
                self.hotspot_loc = mu
            else:
                mu = self.hotspot_loc
            r = 5*np.random.rand(N_hotspots, 1) + 2
            A = np.random.rand(N_hotspots,1)
            X = np.arange(self.mapsize[0])
            Y = np.arange(self.mapsize[1])
            XX, YY = np.meshgrid(X,Y)

            P = np.array([XX.reshape((-1)), YY.reshape(-1)]).T

            D = np.zeros(N)

            for h in range(N_hotspots):
                print("h",h)
                print("mu",mu.shape)
                inner = (repmat(mu[h,:], N, 1) - P)**2
                D = D + A[h].item() * np.exp(-np.sum(inner, axis=1)/(r[h]*r[h]))


            # generate voigts
            mina = sbr / 2
            alpha = mina + mina*np.random.rand(K)
            if self.gamma is None:
                gamma = np.random.gamma(21, 0.5, size=K)
                self.gamma = gamma
            else:
                gamma = self.gamma
            if self.c is None:
                c = gamma + (self.Nw - 2*gamma)*np.random.rand(K)
                c = np.rint(c)

                self.c = c
            else:
                c = self.c
            if self.eta is None:
                eta = np.random.rand(K)
                self.eta = eta
            else:
                eta = self.eta
            

            Vp = self.pseudo_voigt(self.w, c, gamma, eta)

            if plot:
                plt.figure()
                plt.title('Voigt profiles')
                plt.plot(Vp.T)
                plt.show()

            spec = np.sum(Vp, axis=0).T

            if plot:
                plt.figure()
                plt.title('True underlying spectrum')
                plt.plot(self.w, spec)
                plt.show()

            A = repmat(alpha[:,np.newaxis], 1, N).T * repmat(D[:,np.newaxis], 1,  K)
            self.alpha = A

            DD = A @ Vp


        # generate noise
        eta = self.sigma**2 * np.random.randn(N, self.Nw)

        self.real_noise = eta

        self.DD = DD
        self.BB = BB
        self.LL = LL

        X = DD + BB + LL + eta

        self.X = X

        self.map = self.X.reshape(*self.mapsize, -1)

        max_val_map = np.max(self.map)
        self.map_sum = np.zeros_like(self.map[:, :, 0])

        for i, voigt_max in enumerate(self.c):
            idx_max = int(voigt_max)
            map_cut = self.map[:, :, idx_max]
            self.map_sum += map_cut
            if plot:
                plt.matshow(map_cut, vmin=0, vmax=max_val_map)
                plt.colorbar()
                plt.title(f'Map for voight: {i+1}')
                plt.show()
        
        self.map_sum /= len(self.c)
        if plot:
            plt.matshow(self.map_sum)
            plt.colorbar()
            plt.title(f'Map for voights')
            plt.show()

        if plot:
            plt.matshow(X)
            plt.colorbar()
            plt.title('Data matrix')
            plt.show()
        
        if plot_matrix:

            fig, ax = plt.subplots(*self.mapsize, figsize=(20,12))
            N, M = self.mapsize
            for i in range(N):
                for j in range(M):

                    ax[i,j].plot(np.arange(self.Nw), self.map[i,j,:])
                    ax[i,j].set_ylim([0, np.max(self.map)])

                    ax[i,j].grid()
                    fig.tight_layout()
            
            plt.show()

        return X

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            dill.load(f)


if __name__ == '__main__':
    #### PARAMETERS ####
    mapsize = [40, 40]
    W = 100
    seed = 42

    # HOTSPOT LOCATIONS
    hotspot_loc = np.array([
        [10, 3],
        [10, 15],
        [20, 20]
    ])

    # VOIGT LOCATIONS
    c = np.array([25, 75])

    #### GENERATOR
    generator = SERSGenerator(mapsize, W, seed, hotspot_loc=hotspot_loc, c=c)
    X = generator.generate(N_hotspots=len(hotspot_loc), sig=0.1, sbr=2, p_outlier=0, plot=True, plot_matrix=False)
    print(generator.hotspot_loc)
    print(generator.c)

    generator.save("tfh-generator1.pkl")



    

    '''epochs = 5
    for _ in epochs:
        # X has same voight profile in every epoch but noise and background is not the same
        X = generator.generate(N_hotspots=1, K=1, sig=0.2, sbr=1, p_outlier=0, plot=False)
        #model.forward(X)'''

