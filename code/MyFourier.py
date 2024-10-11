import numpy as np
import cv2
import matplotlib.pyplot as plt


# Hàm DFT viết tay (2D Discrete Fourier Transform)
def dft2d(image):
    M, N = image.shape
    dft_image = np.zeros((M, N), dtype=complex)

    for u in range(M):
        for v in range(N):
            sum_value = 0.0
            for x in range(M):
                for y in range(N):
                    exponent = -2j * np.pi * ((u * x / M) + (v * y / N))
                    sum_value += image[x, y] * np.exp(exponent)
            dft_image[u, v] = sum_value

    return dft_image


# Hàm IDFT viết tay (2D Inverse Discrete Fourier Transform)
def idft2d(dft_image):
    M, N = dft_image.shape
    restored_image = np.zeros((M, N), dtype=complex)

    for x in range(M):
        for y in range(N):
            sum_value = 0.0
            for u in range(M):
                for v in range(N):
                    exponent = 2j * np.pi * ((u * x / M) + (v * y / N))
                    sum_value += dft_image[u, v] * np.exp(exponent)
            restored_image[x, y] = sum_value / (M * N)

    return np.abs(restored_image)

if __name__ == '__main__':
    # Đọc ảnh xám
    img = cv2.imread('HoaHong.png', cv2.IMREAD_GRAYSCALE)

    # Tính DFT tay
    dft_image = dft2d(img)

    # Chuyển đổi phổ tần số để hiển thị (logarithmic scaling)
    magnitude_spectrum = np.log(np.abs(dft_image) + 1)

    # Hiển thị phổ tần số DFT
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.axis('off')

    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('DFT Magnitude Spectrum'), plt.axis('off')

    plt.show()

    # Tính IDFT tay để khôi phục ảnh
    restored_image = idft2d(dft_image)

    # Hiển thị ảnh đã khôi phục
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.axis('off')

    plt.subplot(122), plt.imshow(restored_image, cmap='gray')
    plt.title('Restored Image from IDFT'), plt.axis('off')

    plt.show()
