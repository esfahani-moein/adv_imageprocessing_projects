import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json
import os

def get_spectrum_points(magnitude_spectrum):
    """Interactive function to get points from spectrum and save them"""
    plt.figure(figsize=(12, 12))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Click on 8 bright spots in the spectrum\nClose window when done')
    points = plt.ginput(8, timeout=-1)  # Get 8 points from user clicks
    plt.close()
    
    # Convert points to the format we need
    notch_points = [(int(y-magnitude_spectrum.shape[0]//2), int(x-magnitude_spectrum.shape[1]//2)) 
                    for x, y in points]
    
    # Save points to JSON file
    points_dict = {
        'notch_points': notch_points,
        'image_shape': magnitude_spectrum.shape
    }
    
    with open('notch_points.json', 'w') as f:
        json.dump(points_dict, f, indent=4)
    
    print(f"Notch points saved to notch_points.json")
    return notch_points

def load_spectrum_points():
    """Load previously saved spectrum points"""
    if os.path.exists('notch_points.json'):
        with open('notch_points.json', 'r') as f:
            data = json.load(f)
            return data['notch_points']
    return None

def butterworth_notch_reject(shape, d0, uk, vk, n):
    """Creates Butterworth notch reject filter"""
    M, N = shape
    H = np.ones((M, N))
    
    u = np.arange(M)
    v = np.arange(N)
    U, V = np.meshgrid(u-M//2, v-N//2, indexing='ij')
    
    Dk_pos = np.sqrt((U - uk)**2 + (V - vk)**2)
    Dk_neg = np.sqrt((U + uk)**2 + (V + vk)**2)
    
    H_notch = 1 / (1 + (d0**2 / (Dk_pos * Dk_neg))**n)
    
    return H_notch

# Main execution
if __name__ == "__main__":
    # Read and process image
    image_path = 'images/corvette-moire-pattern.tif'
    img = cv.imread(image_path)
    
    if img is None:
        print("Error: Could not load image.")
        exit()
    
    # Convert to grayscale and float32
    img_gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_float32 = np.float32(img_gry)
    
    # Compute DFT
    dft = cv.dft(img_float32, flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Get magnitude spectrum
    magnitude_spectrum = np.log(1 + cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    
    # Try to load existing points first
    notch_points = load_spectrum_points()
    if notch_points is None:
        # If no saved points exist, get new points from user
        notch_points = get_spectrum_points(magnitude_spectrum)
        print("Selected new notch points")
    else:
        print("Using saved notch points from notch_points.json")
    
    # Filter parameters
    rows, cols = img_gry.shape
    d0 = 5  # Filter width
    n = 2   # Filter order
    
    # Create combined filter
    H = np.ones((rows, cols))
    for uk, vk in notch_points:
        H *= butterworth_notch_reject((rows, cols), d0, uk, vk, n)
    
    # Apply filter
    mask_complex = np.stack([H, H], axis=-1)
    filtered_dft = dft_shift * mask_complex
    
    # Inverse transform
    filtered_shift = np.fft.ifftshift(filtered_dft)
    img_filtered = cv.idft(filtered_shift)
    img_filtered = cv.magnitude(img_filtered[:,:,0], img_filtered[:,:,1])
    img_filtered = cv.normalize(img_filtered, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(img_gry, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.plot([p[1]+cols//2 for p in notch_points], [p[0]+rows//2 for p in notch_points], 'r+')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(H, cmap='gray')
    plt.title('Butterworth Notch Filter')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(img_filtered, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()