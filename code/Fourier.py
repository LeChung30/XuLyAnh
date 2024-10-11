import numpy as np
import cv2
import matplotlib.pyplot as plt

# Phép biến đổi Fourier nhanh (FFT)
def fast_fourier_transform(image):
    # Sử dụng numpy để tính FFT 2D
    dft_image = np.fft.fft2(image)
    # Chuyển dịch phổ tần số (để tần số thấp ở giữa)
    dft_shifted = np.fft.fftshift(dft_image)
    return dft_shifted

# Phép biến đổi Fourier ngược (IDFT)
def inverse_fast_fourier_transform(dft_image):
    # Chuyển dịch ngược phổ tần số
    dft_unshifted = np.fft.ifftshift(dft_image)
    # Sử dụng numpy để tính IDFT 2D
    img_restored = np.fft.ifft2(dft_unshifted)
    return np.abs(img_restored)

# Đọc ảnh
img = cv2.imread('HoaHong.png', cv2.IMREAD_GRAYSCALE)

# Tính FFT của ảnh
fft_image = fast_fourier_transform(img)

# Chuyển đổi để hiển thị phổ tần số (logarithmic scaling)
magnitude_spectrum = np.log(np.abs(fft_image) + 1)

# Hiển thị phổ tần số
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT Magnitude Spectrum'), plt.axis('off')

plt.show()

# Khôi phục ảnh từ FFT bằng IDFT
restored_image = inverse_fast_fourier_transform(fft_image)

# Hiển thị ảnh đã khôi phục
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(122), plt.imshow(restored_image, cmap='gray')
plt.title('Restored Image from IFFT'), plt.axis('off')

plt.show()
