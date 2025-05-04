# Функція для перевірки, чи клітинка знаходиться на дошці
def is_valid(x, y, size=5):
    return 0 <= x < size and 0 <= y < size

# Перетворення координат на індекс
def cell_to_index(x, y, size=5):
    return x * size + y

def generate_adjacency_matrix(size=5):
    moves = [(-2, -1), (-1, -2), (1, -2), (2, -1),
             (2, 1), (1, 2), (-1, 2), (-2, 1)]

    matrix = [[0 for _ in range(size * size)] for _ in range(size * size)]

    for x in range(size):
        for y in range(size):
            from_idx = cell_to_index(x, y, size)
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny, size):
                    to_idx = cell_to_index(nx, ny, size)
                    # Якщо ціль — клітинка з індексом 0, ставимо вагу 100
                    matrix[from_idx][to_idx] = 100 if to_idx == 0 else 1

    matrix[0][0] = 100

    return matrix

if __name__ == '__main__':
    adj_matrix = generate_adjacency_matrix()

    # Друк заголовка
    header = "    " + "".join(f"{i:>4}" for i in range(len(adj_matrix)))
    print(header)

    # Друк рядків матриці з індексами
    for i, row in enumerate(adj_matrix):
        row_str = " ".join(f"{val:>3}" for val in row)
        print(f"{i:>2} | {row_str}")
