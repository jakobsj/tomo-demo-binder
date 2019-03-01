
import numpy as np
import matplotlib.pylab as pylab
import astra

def disp_geometry(proj_geom, vol_geom):
    # Extract parameters
    N = vol_geom['GridRowCount']
    source_pos = proj_geom['DistanceOriginSource']
    det_pos = proj_geom['DistanceOriginDetector']
    det_lim = proj_geom['DetectorCount']*proj_geom['DetectorWidth']/2
    
    # Plot source
    pylab.plot(-source_pos,0,'or',markersize=10)
    
    # Center of rotation position
    pylab.plot(0,0,'ok',markersize=10)
    
    # Detector
    pylab.plot(np.ones((3,1))*det_pos,np.array([[det_lim],[-det_lim],[0]]),'ob',markersize=10)
    pylab.plot(np.ones((2,1))*det_pos,np.array([[-1],[1]])*det_lim,'-b')
    
    # Square object box
    xo = np.array([[1],[-1],[-1], [1],[1]])*N*0.5;
    yo = np.array([[1], [1],[-1],[-1],[1]])*N*0.5;
    pylab.plot(xo,yo,'-k')
    
    # Source to detector limits
    pylab.plot(np.array([[-source_pos],[det_pos]]),np.array([[0],[det_lim]]),'-r')
    pylab.plot(np.array([[-source_pos],[det_pos]]),np.array([[0],[-det_lim]]),'-r')
    pylab.plot(np.array([[-source_pos],[det_pos]]),np.array([[0],[0]]),'-r')
    
    # Customize
    pylab.axis('equal')
    pylab.xlim((-source_pos-0.5*det_pos,det_pos+0.5*det_pos))
    pylab.legend(('X-ray source & beam', \
                  'Object box & center of rotation', \
                  'Detector'), loc='upper left')


def fbp_cpu(sinogram, proj_id):
    
    # Get projection geometry from projector
    proj_geom = astra.projector.projection_geometry(proj_id)
    vol_geom = astra.projector.volume_geometry(proj_id)
    
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
    
    # Create a data object to hold the sinogram data
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram);

    # Create configuration for reconstruction algorithm
    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    
    # Create and run the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    # Get the result
    rec = astra.data2d.get(rec_id)
    
    # Clean up. 
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    
    return rec


def fbp_gpu(sinogram, vol_geom, proj_geom, filter_type='Ram-Lak'):
    
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create( '-vol', vol_geom)
    
    # Create a data object to hold the sinogram data
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    
    # Create configuration for reconstruction algorithm
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    
    # possible values for FilterType:
    # none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    # blackman-nuttall, flat-top, kaiser, parzen
    cfg['FilterType'] = filter_type
    
    # Create and run the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    # Get the result
    rec = astra.data2d.get(rec_id)
    
    # Fix inconsistent scaling and orientation of GPU reconstruction
    cor_fac = proj_geom['DistanceOriginSource'] * \
        (proj_geom['DistanceOriginSource'] + proj_geom['DistanceOriginDetector'])
    rec = cor_fac*np.rot90(rec,2)
    
    # Clean up. 
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    
    return rec

def plot_roi(N, num_detector_pixels):
    theta = np.linspace(0,2*np.pi,1000)
    x = 0.5 + 0.5*N + 0.5*num_detector_pixels*np.cos(theta)
    y = 0.5 + 0.5*N + 0.5*num_detector_pixels*np.sin(theta)
    pylab.plot(x,y,'-r')

def pad_sinogram_roi(sinogram,padsize):
    if padsize > 0:
        sinogram_pad = np.hstack( (np.tile(sinogram[:,0], (padsize,1)).transpose(), \
                    sinogram, \
                    np.tile(sinogram[:,-1],(padsize,1)).transpose()) )
    else:
        sinogram_pad = sinogram
    return sinogram_pad

def crop_sinogram(sinogram, crop_size):
    if crop_size > 0:
        sinogram_cropped = sinogram[:,crop_size:-crop_size]
    else:
        sinogram_cropped = sinogram
    return sinogram_cropped

def pad_sinogram_cor(sinogram, cor_padsize):
    if cor_padsize > 0:
        sinogram_cor = np.hstack( (np.tile(sinogram[:,0], (cor_padsize,1)).transpose(), \
                       sinogram ) )
    else:
        sinogram_cor = np.hstack( (sinogram, \
                       np.tile(sinogram[:,-1], (-cor_padsize,1)).transpose() ) )
    return sinogram_cor
