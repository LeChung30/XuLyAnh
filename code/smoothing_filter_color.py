import numpy as np
import cv2

# Phép nhân chập (convolution) cho ảnh màu
def convolve_color(image, kernel):
    kernel_height, kernel_width = kernel.shape
    img_height, img_width, img_channels = image.shape

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Thêm padding với biên của ảnh
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')

    # Ảnh đầu ra rỗng
    output = np.zeros_like(image)

    # Thực hiện phép nhân chập
    for y in range(img_height):
        for x in range(img_width):
            # Lấy vùng ảnh tương ứng với kernel cho từng kênh màu
            for c in range(img_channels):
                region = padded_image[y:y+kernel_height, x:x+kernel_width, c]
                output[y, x, c] = np.sum(region * kernel)

    return output

# Lọc Trung bình (Mean Filter) cho ảnh màu
def mean_filter_color(img):
    mean_filter = np.ones((3, 3)) / 9
    smoothed_image = convolve_color(img, mean_filter)
    return smoothed_image

# Lọc Gaussian cho ảnh màu
def gaussian_filter_color(img):
    gaussian_filter = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16
    gaussian_smoothed_image = convolve_color(img, gaussian_filter)
    return gaussian_smoothed_image

# Lọc Min cho ảnh màu
def min_filter_color(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                filtered_image[i, j, c] = np.min(region)

    return filtered_image

# Lọc Max cho ảnh màu
def max_filter_color(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                filtered_image[i, j, c] = np.max(region)

    return filtered_image

# Lọc Trung vị (Median Filter) cho ảnh màu
def median_filter_color(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                filtered_image[i, j, c] = np.median(region)

    return filtered_image

# Lọc làm sắc nét (Sharpen Filter) cho ảnh màu
def sharpen_filter_color(img):
    sharpen_filter = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened_image = convolve_color(img, sharpen_filter)
    return sharpened_image

if __name__ == '__main__':
    # Đọc ảnh màu
    img = cv2.imread('AnhNhieu.png')

    # Áp dụng các bộ lọc
    mean = mean_filter_color(img)
    gaussian = gaussian_filter_color(img)
    sharpen = sharpen_filter_color(img)
    min_f = min_filter_color(img, 3)
    max_f = max_filter_color(img, 3)
    median_f = median_filter_color(img, 3)

    # Hiển thị ảnh gốc và các ảnh đã lọc
    cv2.imshow('Original Image', img)
    cv2.imshow('Mean Filter', np.clip(mean, 0, 255).astype(np.uint8))
    cv2.imshow('Gaussian Filter', np.clip(gaussian, 0, 255).astype(np.uint8))
    cv2.imshow('Sharpen Filter', np.clip(sharpen, 0, 255).astype(np.uint8))
    cv2.imshow('Min Filter', np.clip(min_f, 0, 255).astype(np.uint8))
    cv2.imshow('Max Filter', np.clip(max_f, 0, 255).astype(np.uint8))
    cv2.imshow('Median Filter', np.clip(median_f, 0, 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
