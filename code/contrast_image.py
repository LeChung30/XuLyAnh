import cv2
import numpy as np

def adjustContrast(image, value_contrast):
    # Đọc ảnh
    img = cv2.imread(image, cv2.COLOR_RGB2GRAY)  # Đọc ảnh ở chế độ grayscale

    # Tìm giá trị pixel nhỏ nhất và lớn nhất trong ảnh
    min_pixel = np.min(img)
    max_pixel = np.max(img)

    # Khởi tạo ma trận kết quả với cùng kích thước như ảnh gốc
    adjusted = np.zeros(img.shape, np.uint8)
    val = value_contrast * 255 / (max_pixel - min_pixel)

    # Áp dụng công thức Min-Max Scaling để điều chỉnh độ tương phản và nhân với contrast_factor
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_pixel_value = val * (img[i][j] - min_pixel)
            adjusted[i][j] = np.clip(new_pixel_value, 0, 255)

    # Hiển thị ảnh gốc và ảnh sau khi chỉnh sửa
    cv2.imshow("Original Image", img)
    cv2.imshow("Min-Max Contrast Adjusted", adjusted)

    # Đợi người dùng nhấn phím bất kỳ và đóng các cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def negative(image):
    img = cv2.imread(image, cv2.COLOR_RGB2GRAY)
    adjusted = 255-img

    cv2.imshow("Original Image", img)
    cv2.imshow("Min-Max Contrast Adjusted", adjusted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def logarithmic(image, c):
    # Đọc ảnh ở chế độ grayscale
    img = cv2.imread(image, cv2.COLOR_RGB2GRAY)

    # Áp dụng hàm logarithm
    adjusted = c * np.log1p(img)  # np.log1p(x) tính log(1 + x)

    # Chuẩn hóa giá trị về khoảng [0, 255]
    adjusted = np.clip(adjusted, 0, 255)

    # Chuyển đổi kiểu dữ liệu về uint8 để hiển thị
    adjusted = np.uint8(adjusted)

    cv2.imshow("Original Image", img)
    cv2.imshow("Logarithmic", adjusted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def power_law(image, gamma):
    # Đọc ảnh
    img = cv2.imread(image, cv2.COLOR_RGB2GRAY)

    adjusted = 255 * (img / 255) ** gamma
    adjusted = np.clip(adjusted, 0, 255)  # Đảm bảo giá trị trong khoảng [0, 255]

    cv2.imshow("Original Image", img)
    cv2.imshow("Logarithmic", adjusted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    adjustContrast('HoaHong.png', 2)
    # negative('HoaHong.png')
    # logarithmic('HoaHong.png', 75)
    # power_law('HoaHong.png', 1)