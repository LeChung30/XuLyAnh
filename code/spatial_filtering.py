import numpy as np


def generate_matrix(matrix):
    # Chuyển ma trận thành numpy array để dễ dàng thao tác
    matrix = np.array(matrix)

    # Kích thước ma trận gốc
    rows, cols = matrix.shape

    # Tạo ma trận mới với kích thước lớn hơn (bao ngoài bằng 0)
    new_matrix = np.zeros((rows + 2, cols + 2))

    # Sao chép ma trận gốc vào giữa ma trận mới
    new_matrix[1:-1, 1:-1] = matrix

    return new_matrix


def convolve(matrix, kernel):
    # Chuyển ma trận và kernel thành numpy array nếu chưa phải
    matrix = np.array(matrix)
    kernel = np.array(kernel)

    # Tạo ma trận mới với padding (bao ngoài ma trận bằng 0)
    padded_matrix = generate_matrix(matrix)

    # Kích thước ma trận và kernel
    m_rows, m_cols = matrix.shape
    k_rows, k_cols = kernel.shape

    # Ma trận kết quả có kích thước bằng với ma trận gốc
    output = np.zeros((m_rows, m_cols))

    # Thực hiện tích chập
    for i in range(1, m_rows + 1):  # Bắt đầu từ 1 vì đã padding
        for j in range(1, m_cols + 1):
            # Lấy cửa sổ con có cùng kích thước với kernel
            window = padded_matrix[i - 1:i + k_rows - 1, j - 1:j + k_cols - 1]
            # Nhân từng phần tử trong cửa sổ với kernel và cộng lại
            output[i - 1, j - 1] = np.sum(window * kernel)

    return output


# Ma trận I và kernel Hx
I = [[5, 5, 5, 5, 5, 5, 5],
     [4, 5, 5, 5, 5, 5, 5],
     [3, 4, 5, 5, 5, 5, 5],
     [3, 3, 4, 5, 5, 5, 5],
     [3, 3, 3, 4, 4, 4, 4],
     [3, 3, 3, 3, 3, 3, 3],
     [3, 3, 3, 3, 3, 3, 3]]

Hx = [[-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1]]

# Gọi hàm convolve
result = convolve(I, Hx)

# In kết quả
print("Kết quả sau khi tích chập:")
print(result)
