#Q block and Inverse Q block
import os
import pickle
import struct

import numpy as np

from arith_utils import arith_encode, arith_decode
from sq_utils import wavread, wavwrite, quantiz, lloyds, save, load
from lab2_arith import enc_arith, dec_arith
def uniform_sq_video(video_array, bitfile_pattern, hist_file_pattern, codebook_file, Nlevels):
    """
    Uniform scalar quantization and encoding for video frames.
    - video_array: 3D numpy array of shape (num_frames, height, width)
    - bitfile_pattern: Filepath pattern with '{frame}' placeholder to save compressed frame bitstreams
    - hist_file_pattern: Filepath pattern with '{frame}' placeholder to save frame symbol counts
    - codebook_file: Filepath to save the uniform codebook
    - Nlevels: Number of reconstruction levels
    """
    num_frames, height, width = video_array.shape
    xmin, xmax = np.min(video_array), np.max(video_array)
    
    # Design initial uniform quantizer
    stepsize = (xmax - xmin) / Nlevels
    partition = np.arange(xmin + stepsize, xmax - stepsize, stepsize)
    codebook = np.arange(xmin + stepsize/2, xmax - stepsize/2, stepsize)
    
    save(filename=codebook_file, codebook=codebook)
    
    total_bits = 0
    total_mse = 0.0
    
    for f in range(num_frames):
        frame = video_array[f]
        # Quantize the frame (flattened)
        index, xq, MSE = quantiz(frame.flatten(), partition, codebook)
        index = index + 1
        total_mse += MSE
        
        # Create histogram
        counts = []
        for i in range(1, Nlevels + 1):
            current_count = np.sum(index == i)
            if current_count == 0:
                current_count = 1
            counts.append(int(current_count))
            
        hist_file = hist_file_pattern.format(frame=f)
        save(filename=hist_file, counts=counts)
        
        hist_file_pkl = f"{hist_file}.pkl"
        with open(hist_file_pkl, "wb") as fh:
            pickle.dump(counts, fh)
            
        # Encode frame
        index_list = index.astype(int).tolist()
        bitfile = bitfile_pattern.format(frame=f)
        Nbits = enc_arith(index_list, hist_file_pkl, bitfile)
        total_bits += Nbits
        
    avg_mse = total_mse / num_frames
    return total_bits, avg_mse


def inverse_uniform_sq_video(bitfile_pattern, codebook_file, hist_file_pattern, outfile, num_frames, height, width):
    """
    Illustrates how the inverse uniform scalar quantization concept applies to video frames.
    
    For a video, instead of a 1D audio vector, each decoded signal is a reconstructed 2D frame 
    (or 3D YUV/RGB array). This function loops over frames, reconstructs each frame's spatial matrix, 
    and could compile them back into a video file.
    """
    # Load the uniform codebook used for frame quantization
    codebook = load(codebook_file)['codebook']
    
    reconstructed_frames = []
    
    for f in range(num_frames):
        # 1. Decode each frame's arithmetically encoded bitstream
        bitfile = bitfile_pattern.format(frame=f)
        hist_file = hist_file_pattern.format(frame=f)
        
        index_dec = dec_arith(hist_file, bitfile)
        
        # 2. De-quantize the index to recover pixel intensities (or transform coefficients)
        xdq_flat = codebook[np.asarray(index_dec) - 1]
        
        # 3. Reshape the 1D de-quantized array back to 2D frame dimensions (Height x Width)
        frame = xdq_flat.reshape((height, width))
        reconstructed_frames.append(frame)
        
    # Convert list of reconstructed frames into a 3D video array (Frames x Height x Width)
    video_array = np.array(reconstructed_frames)
    
    print(f"Reconstructed video array of shape {video_array.shape} successfully.")
    # In a real pipeline, you would use a tool like cv2.VideoWriter or ffmpeg to save this array as a video file.
    return video_array

def uniform_sq_frame_encode(frame, bitfile, codebook_file, hist_file, Nlevels):
    # 1. Get min and max of the frame
    xmin = np.min(frame)
    xmax = np.max(frame)

    # 2. Create initial uniform quantizer securely with exactly Nlevels
    # (np.arange excludes its end point, unlike MATLAB's a:step:b, so build the
    #  arrays by count: Nlevels-1 thresholds and Nlevels reconstruction levels)
    stepsize = (xmax - xmin) / Nlevels
    partition = xmin + stepsize * np.arange(1, Nlevels)
    codebook = xmin + stepsize * (np.arange(Nlevels) + 0.5)

    # 3. Save the uniform codebook
    save(filename=codebook_file, codebook=codebook)

    # 4. Quantize the flattened frame
    index, xq, MSE = quantiz(frame.flatten(), partition, codebook)
    index = index + 1

    # 5. Count occurrences using NumPy operations
    counts = []
    for i in range(1, Nlevels + 1):
        current_count = np.sum(index == i)
        if current_count == 0:
            current_count = 1
        counts.append(int(current_count))

    save(filename=hist_file, counts=counts)
    hist_pkl = f"{hist_file}.pkl"
    with open(hist_pkl, "wb") as f:
        pickle.dump(counts, f)

    # Convert index array to a Python list of standard ints
    index_list = index.astype(int).tolist()

    # 6. Encode index vector
    Nbits = enc_arith(index_list, hist_pkl, bitfile)

    return Nbits, MSE

def uniform_sq_frame_decode(bitfile, codebook_file, hist_file, height, width):
    hist_pkl = f"{hist_file}.pkl"
    # 1. Decode indices
    index_dec = dec_arith(hist_pkl, bitfile)

    # 2. Load codebook
    codebook = load(filename=codebook_file)['codebook']

    # 3. De-quantize
    xdq_flat = codebook[np.asarray(index_dec) - 1]

    # 4. Reshape to 2D frame
    frame_dec = xdq_flat.reshape((height, width))
    return frame_dec