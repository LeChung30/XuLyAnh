from cv2 import imread, imwrite, cvtColor, COLOR_BGR2RGB
from matplotlib.pyplot import subplot, title, imshow, show
from skimage.color import rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb
import os


def colorConvertHSV(image):
    # Kiểm tra xem tệp ảnh có tồn tại không
    if not os.path.exists(image):
        print(f"File {image} not found.")
        return

    # Đọc ảnh
    i = imread(image)

    # Kiểm tra nếu ảnh được đọc thành công
    if i is None:
        print("Error: Image could not be loaded.")
        return

    # Chuyển đổi từ BGR sang RGB
    i_rgb = cvtColor(i, COLOR_BGR2RGB)

    # Chuyển đổi ảnh từ RGB sang HSV
    hsv = rgb2hsv(i_rgb)

    # Trích xuất các kênh HSV
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Lưu các kênh riêng lẻ dưới dạng hình ảnh grayscale
    imwrite('pexels_h.jpg', (h * 255).astype('uint8'))
    imwrite('pexels_s.jpg', (s * 255).astype('uint8'))
    imwrite('pexels_v.jpg', (v * 255).astype('uint8'))

    # Tạo lại ảnh từ các kênh HSV
    hsv_reconstructed = hsv2rgb(hsv)

    # Hiển thị hình ảnh gốc và ảnh được phục hồi
    subplot(1, 2, 1)
    imshow(i_rgb)
    title("Original")

    subplot(1, 2, 2)
    imshow(hsv_reconstructed)
    title("Reconstructed")

    show()

def colorConvertXYZ(image):
    # Kiểm tra xem tệp ảnh có tồn tại không
    if not os.path.exists(image):
        print(f"File {image} not found.")
        return

    # Đọc ảnh
    i = imread(image)

    # Kiểm tra nếu ảnh được đọc thành công
    if i is None:
        print("Error: Image could not be loaded.")
        return

    # Chuyển đổi từ BGR sang RGB
    i_rgb = cvtColor(i, COLOR_BGR2RGB)

    # Chuyển đổi ảnh từ RGB sang XYZ
    xyz = rgb2xyz(i_rgb)

    # Trích xuất các kênh XYZ
    h = xyz[:, :, 0]
    s = xyz[:, :, 1]
    v = xyz[:, :, 2]

    # Lưu các kênh riêng lẻ dưới dạng hình ảnh grayscale
    imwrite('pexels_h.jpg', (h * 255).astype('uint8'))
    imwrite('pexels_s.jpg', (s * 255).astype('uint8'))
    imwrite('pexels_v.jpg', (v * 255).astype('uint8'))

    # Tạo lại ảnh từ không gian XYZ
    xyz_reconstructed = xyz2rgb(xyz)

    # Hiển thị hình ảnh gốc và ảnh được phục hồi
    subplot(1, 5, 1)
    imshow(i_rgb)
    title("Original")

    subplot(1, 5, 2)
    imshow(xyz_reconstructed)
    title("Reconstructed")

    show()

# Gọi hàm
colorConvertXYZ('HoaHong.png')
