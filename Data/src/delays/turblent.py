from typing import Tuple, Callable, Mapping, Literal, Optional
import numpy as np
import os, scipy
from scipy.spatial.distance import cdist
from ..utils import MeanVar
import warnings

from ..dtype import Data

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = min(os.cpu_count(), 4)
    fft = pyfftw.interfaces.numpy_fft
except:
    fft = np.fft

warnings.filterwarnings('ignore')

class TurblentData(Data):
    def plot(self, save:bool=False) -> None:
        return super().plot('Turblent', 'RdBu_r', 'rad', save)


class Turblent:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self, nslcs:int, resolution:Mapping[str, float], method:Literal['fft', 'cov', 'eig', 'fractal', 'trend']='fractal', 
               turbgrade:float=2.25, turbvar:float=1.0, naslcs:Optional[int]=None) -> TurblentData:
        naslcs = naslcs or nslcs

        # randomly select SLCs which are turbulated by atmospheric artefacts
        idxs = np.random.choice(np.arange(0, nslcs), naslcs, replace=False)

        # 0: Check inputs
        if method not in ['fft', 'cov', 'eig', 'fractal', 'trend']:
            raise Exception(f"'method' must be either 'fft' (for the fourier transform based method), "
                            f" or 'cov' (for the covariance based method). {method} was supplied, so exiting.  ")
        turbs = np.zeros((*self.shape, nslcs), dtype=np.float32)
        for naslc in range(naslcs):
            turbs[..., idxs[naslc]] = MeanVar.apply(self.getMethod(method)(resolution), turbgrade, turbvar)


        return TurblentData(turbs)
    
    def getMethod(self, method:str) -> Callable[[Mapping[str, float]], np.ndarray]:
        """Return the turbulent atmospheres simulated by the chosen method.  
        The chosen method is chosen from ['fft', 'cov', 'eig', 'fractal', 'trend]"""
        return {'fractal' : FractalDelay(self.shape),
                'fft'   : FFTDelay(self.shape),
                'cov'   : CovDelay(self.shape),
                'eig'   : EigDelay(self.shape),
                'trend' : TrendDelay(self.shape)}.get(method, FractalDelay(self.shape))
    

# This module is based on the matlab scripts written by
# Ramon Hanssen, May 2000, available in the following website:
#     http://doris.tudelft.nl/software/insarfractal.tar.gz
# Reference:
#   Hanssen, R. F. (2001), Radar interferometry: data interpretation
# and error analysis, Kluwer Academic Pub, Dordrecht, Netherlands. Chap. 4.7.
class FractalDelay:
    """ A class Simulate an isotropic 2D fractal surface with a power law behavior.
        Parameters:
            - shape  | (int, int) | number of rows and columns.
            - regime | (float, float, float) | cumulative percentage of spectrum covered by a specific beta
                                                e.g., (0.001, 0.999, 1.0) for larger scale turbulence,
                                                    (0.980, 0.990, 1.0) for medium scale turbulence,
                                                    (0.010, 0.020, 1.0) for smallest scale turbulence
            - beta   |(float, float, float)| power law exponents for a 1D profile of the data.
                    e.g., (5./3., 8./3., 2./3.) in equation (4.7.28) from Hanssen (2001) .
                    P_phi(f) =  P2(f/f0) ^ -5/3    for 1.5  <= f0/f <= 50   km       regime[1]-regime[0]
                                P1(f/f0) ^ -8/3    for 0.25 <= f0/f <= 1.5  km       regime[0]
                                P3(f/f0) ^ -2/3    for 0.02 <= f0/f <= 0.25 km       regime[2]-regime[1]
    """
    def __init__(self, shape:Tuple, regime:Tuple[float, float, float]=(0.001, 0.999, 1.00), beta:Tuple[float, float, float]=(5./3., 8./3., 2./3.), **kwargs) -> None:
        self.shape = shape
        self.regime = regime
        self.beta = beta

    def __call__(self, resolution:Mapping[str, float], powscale:float=2e-5, initfreq:float=2e-4, **kwargs) -> np.ndarray:
        """Simulate an isotropic 2D fractal surface with a power law behavior, which cooresponds with the 
            [-5/3, -8/3, -2/3] power law.

            Parameters:
                - resolution | dict | the spatial resolution in x, y directions in meters.
                - powscale | float | multiplier of power spectral density in m^2 e.g. 0.2 cm.
                - initfreq   | float | reference spatial freqency in cycle / m.

            Returns:
                - turbdelay | 2D Array | in size of (length, width) in m.
        """

        beta = np.array(self.beta)
        regime = np.array(self.regime)
        length, width = self.shape

        # simulate a uniform random signal

        noise = np.random.rand(length, width)
        noise = fft.fft2(noise)
        noise = fft.fftshift(noise)


        # scale the spectrum with the power law
        yy, xx = np.mgrid[0:length-1:length*1j,
                      0:width-1:width*1j].astype(np.float32)
        yy -= np.rint(length/2)
        xx -= np.rint(width/2)
        yy *= resolution['y']
        xx *= resolution['x']
        dist = np.asarray(np.sqrt(np.square(xx) + np.square(yy)))    #pixel-wise distance in m

        """ 
        - The power `beta+1` is used as the power exponent is defined for a 1D slice of the 2D spectrum:
            "Adler 1981", shows that the surface profile created by the intersection of a plane and a2-D fractal surface is itself.
             With a fractal dimension equal to that of the 2D surface decreased by one.

        - The power `beta/2` is used because the power spectral density is proportional to the amplitude squared 
            Here, we work with the amplitude, instead of the power. 
            So, we should take sqrt(dist.^beta) = dist.^(beta/2).
        """
        beta = (beta + 1) / 2.

        maxdist = np.max(dist)

        distlimits = (max(regime[0] * maxdist, 4 * np.mean((resolution['x'], resolution['y']))), regime[1] * maxdist)

        indxregime1 = (dist <= distlimits[0])
        indxregime2 = np.multiply(dist >= distlimits[0], dist <= distlimits[1])
        indxregime3 = (dist >= distlimits[1])

        fraction1 = np.power(dist[indxregime1], beta[0])
        fraction2 = np.power(dist[indxregime2], beta[1])
        fraction3 = np.power(dist[indxregime3], beta[2])

        fraction = np.zeros(dist.shape, np.float32)
        fraction[indxregime1] = fraction1
        fraction[indxregime2] = fraction2 / np.min(fraction2) * np.max(fraction[indxregime1])
        fraction[indxregime3] = fraction3 / np.min(fraction3) * np.max(fraction[indxregime2])

        # prevent dividing by zero
        fraction[fraction == 0.] = 1.

        # get the fractal spectrum and transform to spatial domain
        fractal_spectrum = np.divide(noise, fraction)

        turbdelay = self.getInvSpectral(fractal_spectrum)

        # calculate the power spectral density of 1st realization
        initpow = self.getPower(turbdelay, np.mean((resolution['x'], resolution['y'])), initfreq)

        # scale the spectrum to match the input power spectral density.
        fractal_spectrum *= np.sqrt(powscale / initpow)
        turbdelay = self.getInvSpectral(fractal_spectrum)
        return turbdelay
    

    def getPsd(self, data:np.ndarray, resolution:float, initfreq:float=1e-3, display:bool=False) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Get the radially averaged 1D spectrum (power density) of input 2D matrix

            Parameters:
                - data       | 2D Array | displacement in m, free from NaN value.
                - resolution | float | the spatial resolution in m.
                - initfreq   | float | reference spatial freqency in cycle / m.
                - display    | bool | display input data and its calculated 1D power spectrum.

            Returns:
                - freq | 1D Array | frequency in cycle/m
                - psd | 1D Array | the power spectral density in m^2
            
            Reference:
                - Python translation of checkfr.m (Ramon Hanssen, 2000).
        """
        # use the square part of the matrix for spectrum calculation
        squaredata = self.getMaxSquareArray(data)
        dim = squaredata.shape[0]

        # calculate the normalized power spectrum (spectral density)
        datafft = fft.fftshift(fft.fft2(squaredata))
        psd = np.abs(np.multiply(datafft, np.conj(datafft))) / (dim**2)

        # The frequency coordinate in cycle / m
        freq = fft.fftfreq(dim, resolution)
        freq = np.roll(freq, shift=np.ceil((dim-1)/2).astype(int)) #shift 0 to the center

        # calculate the radially average spectrum
        freq, psd = self.getAvgRadialSpectrum(freq, psd)

        initpow, slope = self.getPowerSlope(freq, psd, initfreq)

        if display:
            self.plot_psd(data, initpow, slope, initfreq, freq, psd)

        return initpow, slope, freq, psd
        

    def getAvgRadialSpectrum(self, freq:np.ndarray, psd:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the radially averaged power spectrum

            Parameters:
                - freq | 2D Array | frequency in both x, y direction in size of (dim, dim).
                - psd | 2D Array | power spectral density in size of (dim, dim).

            Returns:
                - freq | 1D Array | frequency in radial direction in size of int(dim/2 - 1).
                - psd | 1D Array | power spectral density in size of int(dim/2 - 1).
        """
        dim = freq.shape[0]
        xfreq, yfreq = np.meshgrid(freq, freq)

        uniquefreq = np.unique(xfreq)[np.unique(xfreq) > 0][0]
        radfreq = np.sqrt(np.square(xfreq) + np.square(yfreq))

        # rotationally averaged 1D spectrum from 2D spectrum

        getIndx = lambda i: np.multiply(radfreq >= (i+0.5)*uniquefreq, radfreq <= (i+1.5)*uniquefreq)
        psd = np.array([np.mean(psd[getIndx(i)]) for i in range(int(dim/2 -1))])

        # Only consider one half of spectrum (due to symmetry)
        freq = np.arange(1, dim/2) * uniquefreq

        return freq, psd
    
    def getPower(self, data:np.ndarray, resolution:float, initfreq:float=1e-3) -> float:
        return self.getPsd(data, resolution, initfreq)[0]

    
    def getPowerSlope(self, freq:np.ndarray, psd:np.ndarray, initfreq:float) -> Tuple[float, float]:
        """ Derive the slope and power of an exponential function in loglog scale
                p = initpower * (freq/initfreq)^(-slope)
                
            Parameters:
                - freq  | ND Array | in cycle / m.
                - psd   | ND Array | for the power spectral density
                - initfreq | float | reference freqency in cycle / m.
        
            Returns:
                - initpow | float | power spectral density at reference frequency in the same unit as the input psd.
                - slope | float | slope of power profile in loglog scale

            Reference:
                - Python translation of pslope.m (Ramon Hanssen, 2000).
        """


        freq = freq.flatten()
        psd = psd.flatten()
        if not np.all(freq != 0.):
            indx = (freq != 0.)
            freq = freq[indx]
            psd = psd[indx]

        # convert to log-log scale
        logfreq = np.log10(freq)
        logpsd = np.log10(psd)

        # fit a linear line
        slope = -1 * np.polyfit(logfreq, logpsd, deg=1)[0]
        
        # interpolate psd at reference frequency
        if initfreq < freq[0] or initfreq > freq[-1]:
            raise ValueError('input frequency of interest {} is out of range ({}, {})'.format(initfreq, freq[0], freq[-1]))
        position = np.interp(np.log10(initfreq), logfreq, range(len(logfreq)))
        loginitpow = np.interp(position, range(len(logpsd)), logpsd)
        initpow = np.power(10, loginitpow)
        return initpow, slope
    
    @staticmethod
    def getMaxSquareArray(data:np.ndarray) -> np.ndarray:
        """Grab the max portion of the input 2D matrix that it's:
        1. square in shape
        2. dimension as a power of 2
        """
        # get max square size in a power of 2
        dim = min(data.shape)
        dim = np.power(2, int(np.log2(dim)))

        # find corner with least number of zero values
        zeroflag = (data != 0)

        if data.shape[0] > data.shape[1]:
            num_top = np.sum(zeroflag[:dim, :dim])
            num_bottom = np.sum(zeroflag[-dim:, :dim])
            if num_top > num_bottom:
                data = data[:dim, :dim]
            else:
                data = data[-dim:, :dim]
        else:
            num_left = np.sum(zeroflag[:dim, :dim])
            num_right = np.sum(zeroflag[:dim, -dim:])
            if num_left > num_right:
                data = data[:dim, :dim]
            else:
                data = data[:dim, -dim:]

        return data
    

    
    @staticmethod
    def getInvSpectral(data:np.ndarray) -> np.ndarray:
        """ get the inverse of the 2D spectral """
        data = fft.ifft2(data)
        data = np.abs(data).astype(np.float32)
        data -= np.mean(data)
        return data

class FFTDelay:
    """A class to create synthetic turbulent troposphere delay using an FFT approach.

    Parameters:
        - shape | (int, int) | number of rows and columns.

    """
    def __init__(self, shape=Tuple[int, int], **kwargs) -> None:
        self.shape = shape

    def __call__(self, resolution:Mapping[str, float], powscale:float=1e-2, **kwargs) -> np.ndarray:
        """simulate the turbulent troposphere delay using an FFT approach. 
        The power of the turbulence is tuned by the weather model at the longer wavelengths.

        Parameters:
            - resolution | dict | the spatial resolution in x, y directions in meters.
            - powscale | float | the scaling of the turbulence power according to the weather model at the longer wavelengths.
        
        Returns:
            - turbdelay | 2D Array | in size of (length, width) in m.
        """
        length, width = self.shape
        freqcutoff = 1/50000                                                                                  # drop wavelengths above 50 km
        xfreq, yfreq = np.arange(0, int(width/2)),  np.arange(0, int(length/2))                                  # positive frequencies only
        xfreq, yfreq = np.divide(xfreq, width * resolution['x']), np.divide(yfreq, length * resolution['y'])
        xfreq, yfreq = np.meshgrid(xfreq, yfreq)

        freq = np.sqrt((np.square(xfreq)  + np.square(yfreq))/2)                                                       # 2D positive frequencies
        logpower = np.log10(freq) * (-11/3)                                                                     # -11/3 in 2D gives -8/3 in 1D
        logpower[np.where(freq < (2/3))] = np.log10(freq[np.where(freq < (2/3))]) * (-8/3) - (np.log10(2/3))    # change slope at 1.5 km (2/3 cycles per km)

        binpower = np.power(10, logpower)
        binpower[np.where(freq < freqcutoff)] = 0

        apspower = np.zeros((length, width))                                                                     # mirror positive frequencies into other quadrants
        apspower[0:int(length/2), 0:int(width/2)]= binpower
        apspower[0:int(length/2), int(np.ceil(width/2)):] = np.fliplr(binpower)
        apspower[int(np.ceil(length/2)):, 0:int(width/2)] = np.flipud(binpower)
        apspower[int(np.ceil(length/2)):, int(np.ceil(width/2)):] = np.fliplr(np.flipud(binpower))
        apspower = np.sqrt(apspower)
        apspower *= (powscale/np.std(apspower))

        # simulate a uniform random signal
        noise = np.random.randn(length, width)                                                                   # white noise
        noise = fft.fft2(noise)
        apsfiltered = np.multiply(noise, apspower)                                                              # convolve with filter

        turbdelay = np.real(fft.ifft2(apsfiltered)).astype(np.float32)
        # turbdelay *= (powscale/np.std(turbdelay))                                                           # adjust the turbulence by the weather model at the longer wavelengths.
        return turbdelay 
    

class CovDelay:
    """A class to create synthetic turbulent troposphere delay using an Cov approach.

    Parameters:
        - shape | (int, int) | number of rows and columns.
        - interplimit | int | if npixs is greater than this, images will be generated at size so that the total number of pixels doesn't exceed this.  
                                        e.g. if set to 1e4 (10000, the default) and images are 120*120, they will be generated at 100*100 then upsampled to 120*120.  
    
    """
    def __init__(self, shape:Tuple[int, int], interplimit:int=1e4, **kwargs) -> None:
        self.shape = shape
        self.interpflag = np.prod(shape) > interplimit
        self.oversizefacior = max(np.sqrt(np.prod(shape) / interplimit), 1)
    
    def __call__(self, resolution:Mapping[str, float], covlength:float=2e3, **kwargs) -> np.ndarray:
        """ simulate the turbulent troposphere delay using an COV approach. given a matrix of pixel distances (in meters) and a length scale for the noise (also in meters),
        generate some 2d spatially correlated noise.
        
        Parameters:
            - resolution | dict | the spatial resolution in x, y directions in meters.
            - covlength | float | Length scale over which the noise is correlated.  units are metres.

        Returns:
            - turbdelay | 2D Array | in size of (length, width) in m.
        """

        length, width = self.shape

        ypixs = int(length/self.oversizefacior)
        xpixs = int(width/self.oversizefacior)

        xdists, ydists = np.meshgrid(np.arange(0, xpixs) * resolution['x'] * (width/xpixs), 
                                     np.arange(0, ypixs) * resolution['y'] * (length/ypixs))
        ydists = np.flipud(ydists)
        xydists = np.hstack((xdists.reshape(-1, 1), ydists.reshape(-1, 1)))

        dists = cdist(xydists, xydists, metric='euclidean') # calcaulte all pixelwise pairs - slow as (pixels x pixels)
        
        # Generate noise covariance matrix
        covmatrix = np.exp((-1 * dists) / covlength) 
        try:
            matdecomp = np.linalg.cholesky(covmatrix)           # ie covmat = lower_matcomp @ lower_matcomp.T. So, Worse error messages, so best called in a try/except form.
        except:
            matdecomp = scipy.linalg.cholesky(covmatrix, lower=True)     # better error messages than the numpy versio, but can cause crashes on some machines
        
        noise = np.random.randn((ypixs*xpixs))                            # Parsons 2007 syntax for uncorrelated noise
        noise = np.matmul(matdecomp, noise)                              # for correlated noise

        turbdelay = np.reshape(noise, (ypixs, xpixs))

        if self.interpflag:
            func = scipy.interpolate.interp2d(np.arange(0, xpixs), np.arange(0, ypixs), turbdelay, kind='linear')               # interpolate the delay to a larger size.  First we give it meshgrids and values for each point
            turbdelay = func(np.linspace(0, xpixs, width), np.linspace(0, ypixs, length)).astype(np.float32)

        turbdelay -= turbdelay.mean()
        turbdelay *= 1e-2
        return turbdelay
    
class EigDelay:
    """A class to create synthetic turbulent troposphere delay using an eig approach.

    Parameters:
        - shape | (int, int) | number of rows and columns.
        - interplimit | int | if npixs is greater than this, images will be generated at size so that the total number of pixels doesn't exceed this.  
                                        e.g. if set to 1e2 (100, the default) and images are 120*120, they will be generated at 10*10 then upsampled to 120*120.  
    
    """
    def __init__(self, shape:Tuple[int, int], interplimit:int=1e2, **kwargs) -> None:
        self.shape = shape
        self.interpflag = np.prod(shape) > interplimit
        self.oversizefacior = max(np.sqrt(np.prod(shape) / interplimit), 1)
    
    def __call__(self, resolution:Mapping[str, float], covlength:float=2e3, **kwargs) -> np.ndarray:
        """ simulate the turbulent troposphere delay using an Eig approach. 
        given a matrix of pixel distances (in meters) and a length scale for the noise (also in meters),
        generate some 2d spatially correlated noise.
        
        Parameters:
            - resolution | dict | the spatial resolution in x, y directions in meters.
            - covlength | float | Length scale over which the noise is correlated.  units are metres.

        Returns:
            - turbdelay | 2D Array | in size of (length, width) in m.
        """

        length, width = self.shape

        ypixs = int(length/self.oversizefacior)
        xpixs = int(width/self.oversizefacior)

        xdists, ydists = np.meshgrid(np.arange(0, xpixs) * resolution['x'] * (width/xpixs), 
                                     np.arange(0, ypixs) * resolution['y'] * (length/ypixs))
        
        ydists = np.flipud(ydists)
        
        ydists, xdists = np.ravel(ydists), np.ravel(xdists)

        # Generate noise covariance matrix
        covmatrix = np.eye(ypixs*xpixs)
        for indx in range(ypixs*xpixs):
            xtmp = xdists[indx+1:]
            ytmp = ydists[indx+1:]
            xtmp = np.subtract(xdists[indx], xtmp)
            ytmp = np.subtract(ydists[indx], ytmp)
            tmp = np.linalg.norm((xtmp, ytmp), axis=0)
            covmatrix[indx, indx+1:] = np.power(10, -tmp/covlength)    # change distance to become in meters
            covmatrix[indx+1:, indx] = covmatrix[indx, indx+1:]

        eigvalues, eigvector = np.linalg.eig(covmatrix)
        eigvalues = np.diag(eigvalues)
        noise = np.random.randn((ypixs*xpixs))                            # Parsons 2007 syntax for uncorrelated noise
        noise = np.matmul(np.matmul(eigvector, np.sqrt(eigvalues)), noise)
        noise = np.real(noise)

        turbdelay = np.reshape(noise, (ypixs, xpixs))

        if self.interpflag:
            func = scipy.interpolate.interp2d(np.arange(0, xpixs), np.arange(0, ypixs), turbdelay, kind='linear')               # interpolate the delay to a larger size.  First we give it meshgrids and values for each point
            turbdelay = func(np.linspace(0, xpixs, width), np.linspace(0, ypixs, length)).astype(np.float32)

        turbdelay -= turbdelay.mean()

        turbdelay *= 1e-2
        return turbdelay
    
class TrendDelay:
    def __init__(self, shape:Tuple[int, int], beta:float=2.67) -> None:
        self.shape = shape
        self.beta = beta

    def __call__(self, resolution:Mapping[str, float]) -> np.ndarray:
        beta = 8 - 2 * self.beta

        # scale the spectrum with the power law
        # avoid zero distance
        x = 0.25 + np.arange(-self.shape[1]/2, self.shape[1]/2) * resolution['x']
        y = 0.25 + np.arange(-self.shape[0]/2, self.shape[0]/2) * resolution['y']

        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(np.square(xx) + np.square(yy))

        noemer = dist ** (beta/2)
        noemer[noemer == 0] = 1

        # Simulate a uniform random signal
        noise = np.random.rand(*self.shape)

        noise = fft.fftshift(fft.fft2(noise))

        turbdelay = np.abs(fft.ifft2(noise / noemer))
        return turbdelay