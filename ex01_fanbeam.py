
import astra
import numpy as np
import scipy.io
import matplotlib.pylab as pylab

pylab.gray()

import tools

## Parameters to specify

# Object size N-by-N pixels:
N = 512

# Number of detector pixels
num_detector_pixels = 512

# Number of projection angles.
num_angles = 1024

# Distance source to center-of-rotation, given in numbers of pixels.
source_origin = 1500.0

# Distance detector to center-of-rotation, given in numbers of pixels.
detector_origin = 500.0

# Incident X-ray flux in photons per time unit. Lower=noisier.
flux = 1e3

# Use GPU (True) or CPU (False)
do_gpu = False

# Set filter for FBP - only works on GPU. Possible values:
# none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
# triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
# blackman-nuttall, flat-top, kaiser, parzen
filter_type = 'ram-lak'

## Derived parameters

# Geometric magnification: Scaling factor mapping one object pixel in the
# center-of-rotation plane on to one detector pixel.
magnification = (source_origin+detector_origin)/source_origin

# Size of a detector pixel relative to object pixel
detector_pixel_size = magnification

# Angles at which projections are taken. Here equally spaced, 360 degrees.
angles = np.linspace(0,2*np.pi,num_angles,False)

## Set up ASTRA volume geometry

# vol_geom = astra.create_vol_geom(row_count, col_count)
#  Create a 2D volume geometry.  See the API for more information.
#  row_count: number of rows.
#  col_count: number of columns.#%
#  vol_geom: Python dict containing all information of the geometry.

vol_geom = astra.create_vol_geom(N,N)

## Set up ASTRA projection geometry

#  proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, \
#                       angles, source_origin, origin_det)
# 
#  Create a 2D flat fan beam geometry.  See the API for more information.
#  det_width: distance between two adjacent detectors
#  det_count: number of detectors in a single projection
#  angles: projection angles in radians, should be between -pi/4 and 7pi/4
#  source_origin: distance between the source and the center of rotation
#  origin_det: distance between the center of rotation and the detector array
#
#  proj_geom: Python dict containing all information of the geometry

proj_geom = astra.create_proj_geom( \
    'fanflat', \
    detector_pixel_size, \
    num_detector_pixels, \
    angles, \
    source_origin, \
    detector_origin)

## Set up ASTRA projector required for reconstruction on CPU

#  proj_id = astra.create_projector(type, proj_geom, vol_geom)
#  
#  Create a new projector object based on projection and volume geometry.  
#  Used when the default values of each projector are sufficient.  
# 
#  type: type of the projector.  'line_fanflat', 'strip_fanflat'   See API for more information.
#  proj_geom: MATLAB struct containing the projection geometry.
#  vol_geom: MATLAB struct containing the volume geometry.
#
#  proj_id: identifier of the projector as it is now stored in the astra-library.
  
proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

## Display geometry
pylab.figure(1)
pylab.clf()
tools.disp_geometry(proj_geom, vol_geom)
pylab.title('Fan-beam scan geometry')

## Generate test image and sinogram data
X0 = scipy.io.loadmat('phantom512.mat')['X']
sinogram_id, sinogram = astra.create_sino(X0, proj_id)

# Add noise
if ~np.isinf(flux):
    sinogram_noisy = astra.add_noise_to_sino(sinogram,flux)
else:
    sinogram_noisy = sinogram

## Reconstruct using FBP
if do_gpu:
    rec = tools.fbp_gpu(sinogram_noisy, vol_geom, proj_geom, filter_type)
else:
    rec = tools.fbp_cpu(sinogram_noisy, proj_id)


pylab.figure(2)
pylab.imshow(sinogram)
pylab.title('Clean sinogram data')

pylab.figure(3)
pylab.imshow(sinogram_noisy)
pylab.title('Noisy sinogram data')

pylab.figure(4)
pylab.imshow(X0)
pylab.title('Original image')

pylab.figure(5)
pylab.imshow(rec)
pylab.title('FBP reconstruction')

pylab.show()
