import numpy as np
from scipy.special import i0
from scipy.stats import rv_continuous
from warnings import warn
from halotools.empirical_models.ia_models.ia_model_components import alignment_strength

class VonMisesHalf(rv_continuous):
    r"""
    The von Mises distribution adjusted to span the domain (0,pi).
    The Original von Mises PDF is described by ( exp( kappa * cos(x - mu) ) ) / ( 2 pi BesselI_0(kappa) )
    While this version adjusts that to ( exp( kappa * cos( 2*(x - mu) ) ) ) / ( pi BesselI_0(kappa) )
    This adjusts the PDF to reach a full period on the domain (0,pi)
    And preserved the normalization such that integrating over that domain results in
    an intergal of 1.

    This differs from the scipy implementation of von Mises in that it
    is defined for all kappa, not restricted to kappa > 0.

    Note: the von Mises distribution used here cannot adjust all the way
    to the extremes of creating delta functions at 0 and pi, or at pi/2.
    As such, _cap_alignment_strength has been written to cap
    all kappa values within that range.
    """
    
    def _argcheck(self, kappa):
        kappa = np.asarray(kappa)
        
        return kappa == kappa
    
    def _cap_alignment_strength(self, kappa):
        kappa = np.atleast_1d(kappa)
        cap = 0.999

        high = kappa < alignment_strength(cap)
        low = kappa > alignment_strength(-cap)

        kappa[high] = alignment_strength(cap)
        kappa[low] = alignment_strength(-cap)

        return kappa
    
    def _pdf(self, x, kappa):
        # process arguments
        mu = np.pi/2
        kappa = self._cap_alignment_strength(kappa)
        x = np.atleast_1d(x).astype(np.float64)
        
        # TODO: Any more processing here?
        return ( np.exp( kappa * np.cos( 2*(x-mu) ) ) ) / ( np.pi * i0(kappa) )
    
    def _rvs(self, kappa, size=None, max_iter=100, random_state=None):
        
        kappa = self._cap_alignment_strength(kappa)
        if size is None or size == ():
            size = len(kappa)
        if size != 1:
            # If size is an int, the first condition must be met, if size is a tuple, the second condition is the equivalent form
            if len(kappa) == size or kappa.shape == size:
                pass
            elif len(kappa) == 1:
                kappa = np.ones(size)*kappa
            else:
                msg = ('if `size` argument is given, len(kappa) must be 1 or equal to size.')
                raise ValueError(msg)
        else:
            size = len(kappa)
            
        # vector to store random variates
        result = np.zeros(size)

        # take care of kappa=0 case
        zero_kappa = (kappa == 0)
        uran0 = random_state.random(np.sum(zero_kappa))*np.pi
        result[zero_kappa] = uran0
        
        # take care of edge cases, i.e. |kappa| very large or very small
        with np.errstate(over='ignore'):
            x = np.exp(kappa)
        edge_mask = ((x == np.inf) | (x == 0.0))
        result[edge_mask & (kappa > 0)] = random_state.choice([0.0,np.pi], size=np.sum(edge_mask & (kappa > 0)))
        result[edge_mask & (kappa < 0)] = np.pi/2

        # TODO: Work out the rejection sampling here

        # apply rejection sampling technique to sample from pdf
        n_sucess = np.sum(zero_kappa) + np.sum(edge_mask)  # number of sucesessful draws from pdf
        n_iter = 0  # number of sample-reject iterations
        kk = kappa[(~zero_kappa) & (~edge_mask)]  # store subset of k values that still need to be sampled
        mask = np.repeat(False,size)  # mask indicating which k values have a sucessful sample
        mask[zero_kappa] = True

        while (n_sucess < size) & (n_iter < max_iter):
            x_maxes = np.where( kk > 0, np.pi/2, 0 )                # x values for which the PDF will have its maximum value at a given kappa
            y_maxes = self.pdf(x_maxes, kappa=kk)                   # maximum values to draw up to for each uniform pull
            x_draws = np.random.uniform(0, np.pi, size=len(kk))     # Random x values
            y_draws = np.random.uniform(0, y_maxes)                 # Randomly draw y values for each x, up to the maximum possible value for that PDF
            pdf = self.pdf(x_draws, kappa=kk)                       # Get the actual PDF for each drawn x using its respective kappa

            keep = y_draws < pdf                                    # Keep it if the y_draw value is under the PDF curve at the given x_draw

            # count the number of succesful samples
            n_sucess += np.sum(keep)

            # store y values
            result[~mask] = x_draws                                 # Store all x values. Bad values will get overwritten in future steps

            # update mask indicating which values need to be redrawn
            mask[~mask] = keep

            # get subset of k values which need to be sampled.
            kk = kk[~keep]

            n_iter += 1

        if (n_iter == max_iter):
            msg = ('The maximum number of iterations reached, random variates may not be representative.')
            warn(msg)

        return result
    