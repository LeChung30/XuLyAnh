import numpy as np
import cv2

# Phép nhân chập (convolution)
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    img_height, img_width = image.shape

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Thêm padding với biên của ảnh
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Ảnh đầu ra rỗng
    output = np.zeros_like(image)

    # Thực hiện phép nhân chập
    for y in range(img_height):
        for x in range(img_width):
            # Lấy vùng ảnh tương ứng với kernel
            region = padded_image[y:y+kernel_height, x:x+kernel_width]
            # Tính tổng tích phân tử của kernel và vùng ảnh
            output[y, x] = np.sum(region * kernel)

    return output

# Lọc Trung bình (Mean Filter)
def mean_filter(img):
    mean_filter = np.ones((3, 3)) / 9
    smoothed_image = convolve(img, mean_filter)
    return smoothed_image

# Lọc Gaussian
def gaussian_filter(img):
    gaussian_filter = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16
    gaussian_smoothed_image = convolve(img, gaussian_filter)
    return gaussian_smoothed_image

# Lọc Min
def min_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.min(region)

    return filtered_image

# Lọc Max
def max_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.max(region)

    return filtered_image

# Lọc Trung vị (Median Filter)
def median_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.median(region)

    return filtered_image

# Lọc làm sắc nét (Sharpen Filter)
def sharpen_filter(img):
    sharpen_filter = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened_image = convolve(img, sharpen_filter)
    return sharpened_image

if __name__ == '__main__':
    # Đọc ảnh màu và chuyển đổi sang grayscale
    img = cv2.imread('AnhNhieu.png', cv2.IMREAD_GRAYSCALE)

    # Áp dụng các bộ lọc
    mean = mean_filter(img)
    gaussian = gaussian_filter(img)
    sharpen = sharpen_filter(img)
    min_f = min_filter(img, 3)
    max_f = max_filter(img, 3)
    median_f = median_filter(img, 3)

    # Hiển thị ảnh gốc và các ảnh đã lọc
    cv2.imshow('Original Image', img)
    cv2.imshow('Mean Filter', mean.astype(np.uint8))
    cv2.imshow('Gaussian Filter', gaussian.astype(np.uint8))
    cv2.imshow('Sharpen Filter', sharpen.astype(np.uint8))
    cv2.imshow('Min Filter', min_f.astype(np.uint8))
    cv2.imshow('Max Filter', max_f.astype(np.uint8))
    cv2.imshow('Median Filter', median_f.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
