def cell_to_index(x, y, size=5):
    return x * size + y

def generate_init_q_matrix(size=5):
    matrix = [[0 for _ in range(size**2)] for _ in range(size**2)]

    matrix[7][0] = 100
    matrix[11][0] = 100

    return matrix

if __name__ == '__main__':
    q_matrix = generate_init_q_matrix()

    # Друк заголовка
    header = "    " + "".join(f"{i:>4}" for i in range(len(q_matrix)))
    print(header)

    # Друк рядків матриці з індексами
    for i, row in enumerate(q_matrix):
        row_str = " ".join(f"{val:>3}" for val in row)
        print(f"{i:>2} | {row_str}")