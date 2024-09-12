import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.matrixlib.defmatrix import matrix
from skimage.color import rgb2gray
from tifffile import imshow


def convert_im2matrix(image):
    # Đọc ảnh
    i = cv2.imread(image)

    # Chuyển từ BGR sang RGB
    i_rgb = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

    # Chuyển đổi ảnh thành ảnh xám
    gray = rgb2gray(i_rgb)

    # Áp dụng ngưỡng để chuyển đổi ảnh xám thành ảnh nhị phân
    bi = gray > 0.6

    # Chuyển đổi ma trận nhị phân từ boolean sang số nguyên (0 và 1)
    binary_matrix = bi.astype(np.uint8)

    return binary_matrix

def encode_matrix(matrix):
    new_matrix = []

    m = len(matrix)
    n = len(matrix[0])
    for i in range(m):
        a_cur = []
        f = matrix[i][0]
        cnt = 1
        for j in range(1, n):
            if matrix[i][j] == f:
                cnt += 1
            else:
                a_cur.append(int(f))
                a_cur.append(cnt)
                f = matrix[i][j]  # Gán giá trị hiện tại làm phần tử tiếp theo
                cnt = 1
        # Thêm phần tử cuối cùng sau khi kết thúc vòng lặp
        a_cur.append(int(f))
        a_cur.append(cnt)
        new_matrix.append(a_cur)

    return new_matrix

def decode_matrix(matrix):
    decoded_matrix = []
    for i in matrix:
        decoded_row = []
        n = len(i)
        for j in range(0, n, 2):
            value = i[j]
            count = i[j + 1]
            decoded_row.extend([value] * count)
        decoded_matrix.append(decoded_row)

    return decoded_matrix

def convert_ma2im(matrix):
    # Chuyển đổi các giá trị 0 và 1 trong ma trận thành 0 và 255 để hiển thị như ảnh nhị phân
    img = (matrix * 255).astype(np.uint8)

    cv2.imshow("Decoded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

arr_quad_tree = []

def quad_tree(matrix, x1, x2, y1, y2, local, cnt):
    if x1 > x2 or y1 > y2:
        return

    # Nếu vùng chỉ có một pixel
    if x1 == x2 and y1 == y2:
        arr_quad_tree.append((local, cnt, np.uint8(matrix[x1][y1])))
        return

    x_mid = (x1 + x2) // 2
    y_mid = (y1 + y2) // 2

    # Kiểm tra nếu tất cả các phần tử trong vùng đều giống nhau
    s = matrix[x1][y1]
    uniform = True

    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            if matrix[i][j] != s:
                uniform = False
                break
        if not uniform:
            break

    # Nếu tất cả các phần tử giống nhau, thêm nút vào cây
    if uniform:
        arr_quad_tree.append((local, cnt, np.uint8(s)))
    else:
        # Chia làm 4 phần và đệ quy cho từng phần
        quad_tree(matrix, x1, x_mid, y1, y_mid, 1, cnt + 1)
        quad_tree(matrix, x_mid + 1, x2, y1, y_mid, 2, cnt + 1)
        quad_tree(matrix, x1, x_mid, y_mid + 1, y2, 3, cnt + 1)
        quad_tree(matrix, x_mid + 1, x2, y_mid + 1, y2, 4, cnt + 1)

if __name__ == '__main__':
    m_img = convert_im2matrix("HoaHong.png")
    # matrix_convert = encode_matrix(m_img)
    #
    # decoded_matrix = decode_matrix(matrix_convert)
    #
    # # Chuyển đổi lại ma trận thành hình ảnh và hiển thị
    # convert_ma2im(np.array(decoded_matrix))

    quad_tree(m_img, 0, m_img.shape[0] - 1, 0, m_img.shape[1] - 1, 1, 0)
    print(arr_quad_tree)
