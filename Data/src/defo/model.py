from typing import Tuple, Callable
import numpy as np
from ..utils import MinMax

class Cone:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self) -> np.ndarray:
        x, y = np.meshgrid(np.linspace(-3, 3, self.shape[1]), np.linspace(-3, 3, self.shape[0]))

        z = np.sqrt(np.power(x, 2) + np.power(y, 2))

        z = MinMax.apply(z, -0.75 * np.max(z), np.max(z)) # * np.random.rand()
        
        return z
    
class Peak:
    def __init__(self, shape:Tuple[int, int]) -> None:
        self.shape = shape

    def create(self) -> np.ndarray:
        x, y = np.meshgrid(np.linspace(-3, 3, self.shape[1]), np.linspace(-3, 3, self.shape[0]))

        z = 3 * np.power(1 - x, 2) * np.exp( -np.power(x, 2) - np.power(y + 1 , 2)) \
                - 10 * (x / 5 - np.power(x, 3) - np.power(y, 5)) * np.exp( -np.power(x, 2) - np.power(y, 2)) \
                -1/3 * np.exp( -np.power(x + 1, 2) - np.power(y, 2))

        return z

class Mogi:
    def __init__(self, nu:float=0.25) -> None:

        self.nu = nu

    def __call__(self, center:Tuple[float, float], coord:np.ndarray, depth:float, volume_change:float) -> np.ndarray:
        """ Computes the deformation map in the east-north-up directions 
        
        Parameters:
            - center | (float, float) | Center of the sensource (x, y) in meters.
            - coord | NDArray | Coordinates of the map in meters.
            - depth | float | Depth of the source in meters.
            - volume_change |float | Volume change of the mogi event in m^3.

        Returns:
            - defomap | NDArray |:Displacement map
        """

        disp = np.zeros((3, coord.shape[1]), dtype=np.float32)
        
        east = coord[0, :] - center[0]
        north = coord[1, :] - center[1]
        dist = np.sqrt((np.square(east) + np.square(north) + np.square(depth)))
        
        
        const = (self.nu - 1.) * volume_change / np.pi
        const *= np.power(dist, -3)

        disp[0, ] = np.multiply(east, const)
        disp[1, ] = np.multiply(north, const)
        disp[2, ] = np.multiply(-1.0 * depth, const)


        return disp
    

class Okada:
    def __init__(self, nu:float=0.25) -> None:
        self.nu = nu

    def __call__(self, center:Tuple[float, float], coord:np.ndarray, length:float, width:float, depth:float, slip:float, strike:float, rake:float, dip:float, opening:float) -> np.ndarray:
        """
        OKADA: Surface deformation due to a finite rectangular source.
        computes displacements, tilts and strains at the surface of an elastic
        half-space, due to a dislocation defined by RAKE, SLIP, and OPEN on a
        rectangular fault defined by orientation STRIKE and DIP, and size LENGTH and
        WIDTH. The fault centroid is located (e_center,n_center,-DEPTH).

        Inputs:
            e,n                | rank 2 array | coordinates of observation points in a geographic referential
                                                (East,North,Up) (units are described below)
            e_center, n_center | float        | longitude and latitude of centre of deformation source.
            depth              | float        | depth of the fault centroid (depth > 0)
            strike             | float        | fault trace direction [0 : 2π] (0 to 360° measured clockwise relative to
                                                North), defined so that the fault dips to the right side of the trace
            dip                | float        | angle between the fault and a horizontal plane [0 : π] (0 to 90°)
            length             | float        | fault length in the STRIKE direction (LENGTH > 0)
            width              | float        | fault width in the DIP direction (WIDTH > 0)
            rake               | float        | direction the hanging wall moves during rupture, measured relative
                                                to the fault STRIKE (-180 to 180°).
            slip               | float        | dislocation in RAKE direction (length unit)
            opening            | float        | dislocation in tensile component (same unit as SLIP)
            nu                 | float        | Poisson's ratio
            Plot_flag          | boolean      | if true, produces a 3-D figure with fault geometry and dislocation at
                                                scale.

        Returns:
            x_grid             | rank 2 array | displacement in x direction for each point (pixel on Earth's surface)
            y_grid             | rank 2 array | displacement in y direction for each point (pixel on Earth's surface)
            z_grid             | rank 2 array | displacement in z direction for each point (pixel on Earth's surface)
                                                "units as slip and opening"

        References:
            Aki K., and P. G. Richards, Quantitative seismology, Freemann & Co, New York, 1980.
            Okada Y., Surface deformation due to shear and tensile faults in a half-space,
            Bull. Seismol. Soc. Am., 75:4, 1135-1154, 1985.

        Remarks:
            1. Formulas and notations from Okada [1985] solution excepted for strain convention (here positive strain means
            compression), and for the fault parameters after Aki & Richards [1980], e.g.:
                DIP=90, RAKE=0   : left lateral (senestral) strike slip
                DIP=90, RAKE=180 : right lateral (dextral) strike slip
                DIP=70, RAKE=90  : reverse fault
                DIP=70, RAKE=-90 : normal fault

            2. Equations are all vectorized excepted for argument DIP which must be a scalar (because of a singularity in
            Okada's equations). All other arguments can be scalar or matrix of the same size.

        History:
            2020/09/06         | eng          | written from the original function by François Beauducel <beauducel@ipgp.fr>
                                                "https://github.com/IPGP/deformations-matlab.git"
        """

        disp = np.zeros((3, coord.shape[1]), dtype=np.float32)

        # Defines dislocation in the fault plane system

        # Converts fault coordinates (E,N,DEPTH) relative to centroid into Okada's reference system (X,Y,D)
        d = depth + np.sin(dip) * width/2                              # d is fault's top edg
        ec = coord[0,] - center[0] + np.cos(strike) * np.cos(dip) * width/2     # relative to fault centroid and axis transformation
        nc = coord[1,] - center[1] - np.sin(strike) * np.cos(dip) * width/2     # relative to fault centroid and axis transformation

        x = np.cos(strike) * nc + np.sin(strike) * ec + length/2
        y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * width

        # Variable substitution (independent from xi and eta)
        p = y * np.cos(dip) + d * np.sin(dip)
        q = y * np.sin(dip) - d * np.cos(dip)

        # Displacements (strike-slip, dip-slip, tensile-fault)

        ux = (- np.cos(rake) * slip * self.chinnery(self.ux_ss, x, p, q, length, width, dip, self.nu) / (2 * np.pi)\
            - np.sin(rake) * slip * self.chinnery(self.ux_ds, x, p, q, length, width, dip, self.nu) / (2 * np.pi)\
            + opening * self.chinnery(self.ux_tf, x, p, q, length, width, dip, self.nu) / (2 * np.pi))

        uy = (- np.cos(rake) * slip * self.chinnery(self.uy_ss, x, p, q, length, width, dip, self.nu) / (2 * np.pi)\
            - np.sin(rake) * slip * self.chinnery(self.uy_ds, x, p, q, length, width, dip, self.nu) / (2 * np.pi)\
            + opening * self.chinnery(self.uy_tf, x, p, q, length, width, dip, self.nu) / (2 * np.pi))

        uz = (- np.cos(rake) * slip * self.chinnery(self.uz_ss, x, p, q, length, width, dip, self.nu) / (2 * np.pi)\
            - np.sin(rake) * slip * self.chinnery(self.uz_ds, x, p, q, length, width, dip, self.nu) / (2 * np.pi)\
            + opening * self.chinnery(self.uz_tf, x, p, q, length, width, dip, self.nu) / (2 * np.pi))

        disp[0,] = np.sin(strike) * ux - np.cos(strike) * uy
        disp[1,] = np.cos(strike) * ux + np.sin(strike) * uy
        disp[2,] = uz

        return disp
    
    # Chinnery's notation [equation (24) p. 1143]
    def chinnery(self, func:Callable[[np.ndarray, np.ndarray, np.ndarray, float, float, float, float], np.ndarray], x:np.ndarray, p:np.ndarray, q:np.ndarray, length:float, width:float, dip:float, nu:float) -> np.ndarray:
        """
        Formula to add the different fault components (for more information, see
        Okada, Surface deformation due to shear and tensile faults in a half-space,
        Bulletin of the Seismological Society of America (1985) 75 (4): 1135-1154)
        """
        u = func(x, p, q, dip, nu)\
            - func(x, p - width, q, dip, nu)\
            - func(x - length, p, q, dip, nu)\
            + func(x - length, p - width, q, dip, nu)
        return u
    

    # Displacement sub-functions
    # strike-slip displacement sub-functions [equation (25) p. 1144]
    def ux_ss(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        return xi * q / (r * (r + eta)) + np.arctan(xi * eta / (q * r)) + self.I1(xi, eta, q, dip, nu, r) * np.sin(dip)


    def uy_ss(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        return yb * q / (r * (r + eta)) + q * np.cos(dip) / (r + eta) + self.I2(xi, eta, q, dip, nu, r) * np.sin(dip)


    def uz_ss(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        return db * q / (r * (r + eta)) + q * np.sin(dip) / (r + eta) + self.I4(xi, eta, q, dip, nu, r) * np.sin(dip)


    # dip-slip displacement sub-functions [equation (26) p. 1144]
    def ux_ds(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        return q / r - self.I3(xi, eta, q, dip, nu, r)*np.sin(dip)*np.cos(dip)


    def uy_ds(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        return yb * q / (r * (r + xi)) + np.cos(dip) * np.arctan(xi * eta / (q * r)) - self.I1(xi, eta, q, dip, nu, r) * np.sin(dip) * np.cos(dip)


    def uz_ds(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        return db * q / (r * (r + xi)) + np.sin(dip) * np.arctan(xi * eta / (q * r)) - self.I5(xi, eta, q, dip, nu, r) * np.sin(dip) * np.cos(dip)


    # tensile fault displacement sub-functions [equation (27) p. 1144]
    def ux_tf(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        return (q**2) / (r * (r + eta)) - self.I3(xi, eta, q, dip, nu, r) * np.sin(dip)**2


    def uy_tf(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float) -> np.ndarray:
        r = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta * np.sin(dip) - q * np.cos(dip)
        return -db * q / (r * (r + xi)) - np.sin(dip) * (xi * q / (r * (r + eta)) - np.arctan(xi * eta / (q * r))) - self.I1(xi, eta, q, dip, nu, r) * np.sin(dip) ** 2


    def uz_tf(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float):
        r = np.sqrt(xi**2 + eta**2 + q**2)
        yb = eta * np.cos(dip) + q * np.sin(dip)
        return yb * q / (r * (r + xi)) + np.cos(dip) * (xi * q / (r * (r + eta)) - np.arctan(xi * eta / (q * r))) - self.I5(xi, eta, q, dip, nu, r) * np.sin(dip)**2


    def I1(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float, r:np.ndarray) -> np.ndarray:
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > 10E-8:
            return (1 - 2 * nu) * (-xi / (np.cos(dip) * (r + db))) - self.I5(xi, eta, q, dip, nu, r) * np.sin(dip) / np.cos(dip)
        else:
            return -((1 - 2 * nu) / 2.) * (xi * q / ((r + db) ** 2))


    def I2(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float, r:np.ndarray) -> np.ndarray:
        return (1 - 2 * nu) * (-np.log(r + eta)) - self.I3(xi, eta, q, dip, nu, r)


    def I3(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float, r:np.ndarray) -> np.ndarray:
        yb = eta * np.cos(dip) + q * np.sin(dip)
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > 10E-8:
            return (1 - 2 * nu) * (yb / (np.cos(dip) * (r + db)) - np.log(r + eta)) + self.I4(xi, eta, q, dip, nu, r) * np.sin(
                dip) / np.cos(dip)
        else:
            return ((1 - 2 * nu) / 2.) * (eta / (r + db) + yb * q / ((r + db) ** 2) - np.log(r + eta))


    def I4(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float, r:np.ndarray) -> np.ndarray:
        db = eta * np.sin(dip) - q * np.cos(dip)
        if np.cos(dip) > 10E-8:
            return (1 - 2 * nu) * (np.log(r + db) - np.sin(dip) * np.log(r + eta)) / np.cos(dip)
        else:
            return -(1 - 2 * nu) * q / (r + db)


    def I5(self, xi:np.ndarray, eta:np.ndarray, q:np.ndarray, dip:float, nu:float, r:np.ndarray) -> np.ndarray:
        db = eta * np.sin(dip) - q * np.cos(dip)
        X = np.sqrt(xi ** 2 + q ** 2)
        if np.cos(dip) > 10E-8:
            return (1 - 2 * nu) * 2 * np.arctan((eta * (X + q * np.cos(dip)) + X * (r + X) * np.sin(dip)) / (
                        xi * (r + X) * np.cos(dip))) / np.cos(dip)
        else:
            return -(1 - 2 * nu) * xi * np.sin(dip) / (r + db)
