from Adjacency_matrix_generator import *
from Q_matrix_generator import generate_init_q_matrix
import random

# Розміри поля
size = 5

# Список можливих рухів коня (переміщення на шаховій дошці)
moves = [(-2, -1), (-1, -2), (1, -2), (2, -1),
         (2, 1), (1, 2), (-1, 2), (-2, 1)]


# Функція для перевірки, чи клітинка знаходиться на дошці
def is_valid(x, y, size=5):
    return 0 <= x < size and 0 <= y < size


# Перетворення координат на індекс
def cell_to_index(x, y, size=5):
    return x * size + y


# Виведення шахової дошки
def print_board(knight_pos, size=5):
    board = [['.' for _ in range(size)] for _ in range(size)]
    knight_x, knight_y = knight_pos
    board[knight_x][knight_y] = 'K'  # Позначаємо позицію коня
    for row in board:
        print(" ".join(row))


# Функція для вибору випадкового ходу з поточного стану
def choose_move(curr_pos, R, Q, chosen_moves, size=5):
    x, y = curr_pos
    current_index = cell_to_index(x, y, size)

    # Отримуємо доступні переходи з R
    available_moves = [i for i, open in enumerate(R[current_index]) if open > 0]

    if not available_moves:
        return curr_pos  # якщо немає куди йти, повертаємо поточну позицію

    # Знаходимо хід із максимальною вагою в Q
    max_q = max(Q[current_index][i] for i in available_moves)
    best_moves = [i for i in available_moves if Q[current_index][i] == max_q]
    # Випадковий вибір серед найкращих
    chosen_index = random.choice(best_moves)

    new_x, new_y = divmod(chosen_index, size)
    chosen_moves.append((current_index, chosen_index))
    return (new_x, new_y)

def update_q_matrix(chosen_moves, R, Q):
    gamma = 0.8

    # Знаходимо всі можливі дії з наступного стану (a)
    for s, a in reversed(chosen_moves):
        next_possible_actions = [i for i, r in enumerate(R[a]) if r > -1]

        # Якщо є доступні дії, беремо максимум з Q
        if next_possible_actions:
            max_q_next = max([Q[a][a_next] for a_next in next_possible_actions])
        else:
            max_q_next = 0

        # Оновлюємо Q-значення
        Q[s][a] = R[s][a] + gamma * max_q_next

    return Q

# Додатково: функція для друку дошки з номерами кроків
def print_path_on_board(path, size=5):
    board = [['.' for _ in range(size)] for _ in range(size)]
    for step, (x, y) in enumerate(path):
        board[x][y] = str(step)
    for row in board:
        print(" ".join(cell.rjust(2) for cell in row))

# Стартова позиція
start_pos = (4, 4)
end_pos = (0, 0)

# Ініціалізація
R = generate_adjacency_matrix(size)
Q = generate_init_q_matrix(size)

n_episodes = 10

print("Initial Board:")
print_board(start_pos)

last_path = []  # Збереження шляху останнього епізоду

for episode in range(n_episodes):
    current_pos = start_pos
    chosen_moves = []
    path = [current_pos]  # шлях для відображення
    steps = 0

    while current_pos != end_pos:
        current_pos = choose_move(current_pos, R, Q, chosen_moves)
        path.append(current_pos)
        steps += 1
        if episode == 0:
            print(f"\nStep {steps}:")
            print_board(current_pos)

    print(f"Episode {episode + 1}: finished in {steps} steps")
    update_q_matrix(chosen_moves, R, Q)

    if episode == n_episodes - 1:
        last_path = path  # зберігаємо шлях останнього епізоду

print("\nFinal Q-matrix updated after all episodes.")

# Виведення шляху коня на дошці
print("\nFinal path on board (step numbers):")
print_path_on_board(last_path)