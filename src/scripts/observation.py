import jax.numpy as np
from jax import tree_map
from dLux.observations import AbstractObservation
from dLux import model

class MultiImage(AbstractObservation):
    N : int

    def __init__(self, N):
        super().__init__(name='observation')
        self.N = int(N)
    
    def observe(self, instrument, **kwargs):
        """
        Models the instrument sources through the instrument optics and
        then returns self.N images of the scene with the source flux 
        distributed across the images.
        """
        # if 'individual_psfs' in kwargs and kwargs['individual_psfs']:
        #     source = instrument.scene.sources[0]
        #     wavelengths = source.get_wavelengths()
        #     weights = source.get_weights()
        #     fluxes = source.get_flux()
        #     positions = source.get_position()

        #     psfs = np.array([instrument.optics.propagate_multi(wavelengths, 
        #         positions[i], weights) for i in range(len(fluxes))])
        #     psfs *= fluxes

        #     psfs /= self.N

        psf = model(instrument.optics, sources=instrument.sources)
        psf /= self.N
        if instrument.detector is not None:
            image = instrument.detector.apply_detector(psf)
        else:
            image = psf
        return np.tile(image, (self.N, 1, 1))


class IntegerDither(AbstractObservation):
    dithers : list[int]
    padding : int
    N : int

    def __init__(self, dithers):
        super().__init__(name='observation')
        self.dithers = dithers
        self.N = len(dithers)
        self.padding = int(np.array(dithers).max())
    
    def observe(self, instrument, **kwargs):
        """
        Models the instrument sources through the instrument optics and
        then returns self.N images of the scene with the source flux 
        distributed across the images.
        """
        padded_model = instrument.add('AngularMFT.npixels_out', 
            2*self.padding)
        c = padded_model.AngularMFT.npixels_out//2
        s = instrument.AngularMFT.npixels_out//2

        source = instrument.sources
        psf = padded_model.optics.model(source)
        psfs_out = []
        for i, dither in enumerate(self.dithers):
            x, y = dither
            psfs_out.append(psf[c+y-s:c+y+s, c+x-s:c+x+s]/self.N)

        apply_detector = lambda psf: instrument.detector.apply_detector(psf)
        psfs = np.array(tree_map(apply_detector, psfs_out))
        return psfs