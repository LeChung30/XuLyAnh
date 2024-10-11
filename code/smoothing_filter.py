import numpy as np
import cv2

# Phép nhân chập (convolution)
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    img_height, img_width = image.shape

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # tao ma tran mo rong bien 0 xung quanh
    # padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
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
def mean_filter(img, size=3):
    mean_filter = np.ones((size, size)) / size**2
    smoothed_image = convolve(img, mean_filter)
    return smoothed_image

# Lọc Gaussian
def gaussian_filter(img, kernel_size=3, sigma=1):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    sum_val = 0.0

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i, j]

    # Chuẩn hóa kernel để tổng các phần tử bằng 1
    kernel /= sum_val
    print(kernel)

    gaussian_smoothed_image = convolve(img, kernel)
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

# Lọc làm sắc nét (LapLacian Filter)
def laplacian_filter(img):
    laplacian_filter = np.array([[0, 1, 0],
                               [1, 4, 1],
                               [0, 1, 0]])
    laplacian_image = convolve(img, laplacian_filter)
    return laplacian_image

# Bộ lọc trung điểm cho ảnh xám
def midpoint_filter(image, kernel_size=3):
    pad = kernel_size // 2

    # Thêm padding vào ảnh để xử lý biên
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')

    # Khởi tạo ảnh kết quả
    filtered_image = np.zeros_like(image)

    # Duyệt qua từng pixel trong ảnh
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Trích xuất vùng lân cận pixel hiện tại với kích thước kernel
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Tính giá trị min và max trong vùng
            min_val = np.min(region)
            max_val = np.max(region)
            # Tính giá trị trung điểm
            filtered_image[i, j] = (min_val + max_val) / 2

    return filtered_image

# Bộ lọc trung bình cắt bỏ alpha cho ảnh xám
def alpha_trimmed_mean_filter(image, kernel_size=3, alpha=2):
    pad = kernel_size // 2

    # Thêm padding vào ảnh để xử lý biên
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')

    # Khởi tạo ảnh kết quả
    filtered_image = np.zeros_like(image)

    # Duyệt qua từng pixel trong ảnh
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Trích xuất vùng lân cận pixel hiện tại với kích thước kernel
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Sắp xếp giá trị vùng
            sorted_region = np.sort(region.flatten())

            # Loại bỏ `alpha` giá trị lớn nhất và nhỏ nhất
            trimmed_region = sorted_region[alpha//2 : len(sorted_region) - alpha//2]

            # Tính giá trị trung bình của các giá trị còn lại
            filtered_image[i, j] = np.mean(trimmed_region)

    return filtered_image

if __name__ == '__main__':
    # Đọc ảnh màu và chuyển đổi sang grayscale
    img = cv2.imread('HoaHong.png', cv2.IMREAD_GRAYSCALE)

    # Áp dụng các bộ lọc
    mean = mean_filter(img, 15)
    gaussian = gaussian_filter(img, 15, 10)
    # sharpen = sharpen_filter(img)
    # laplacian = laplacian_filter(img)
    min_f = min_filter(img, 15)
    max_f = max_filter(img, 15)
    median_f = median_filter(img, 15)
    # midpoint_f = midpoint_filter(img, 3)
    # alpha_f = alpha_trimmed_mean_filter(img, 3, 2)

    # Hiển thị ảnh gốc và các ảnh đã lọc
    cv2.imshow('Original Image', img)
    cv2.imshow('Mean Filter', mean.astype(np.uint8))
    cv2.imshow('Gaussian Filter', gaussian.astype(np.uint8))
    # cv2.imshow('Sharpen Filter', sharpen.astype(np.uint8))
    # cv2.imshow('Laplacian Filter', laplacian.astype(np.uint8))
    cv2.imshow('Min Filter', min_f.astype(np.uint8))
    cv2.imshow('Max Filter', max_f.astype(np.uint8))
    cv2.imshow('Median Filter', median_f.astype(np.uint8))
    # cv2.imshow('Midpoint Filter', midpoint_f.astype(np.uint8))
    # cv2.imshow('Alpha Filter', alpha_f.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
