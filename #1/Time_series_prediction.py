import random
import math

# Функція для обчислення значення сигмоїдної функції активації
def sigmoid(x):
    return 1/(1+math.exp(-x))

# Похідна від сигмоїдної функції для використання в back propagation
def sigmoid_derivative(x):
    return x*(1-x)

# Функція для нормалізації даних в діапазоні [0, 1]
def normalize(data, min_value, max_value):
    normalized_data = []
    for value in data:
        if max_value - min_value != 0:
            normalized_value = (value - min_value) / (max_value - min_value)
        else:
            normalized_value = 0
        normalized_data.append(normalized_value)

    return normalized_data

# Функція для денормалізації значення в оригінальний діапазон
def denormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value


raw_data = [
    1.88, 4.52, 1.91, 5.66, 1.23, 5.50, 1.14, 5.29, 1.60, 4.31, 0.06, 5.33, 0.07, 4.62, 0.69
]
test_data = [
    [4.31, 0.06, 5.33],
    [0.06, 5.33, 0.07],
    [5.33, 0.07, 4.62]
]
expected_result = [0.07, 4.62, 0.69]

min_value, max_value = min(raw_data), max(raw_data) # Мінімальне та максимальне значення з оригінальних даних
# Нормалізація всіх даних
data_normalized = normalize(raw_data, min_value, max_value)
test_data = [normalize(test_inp, min_value, max_value) for test_inp in test_data]
expected_result = normalize(expected_result, min_value, max_value)

# Формування навчальних даних з послідовних блоків по три значення
train_x = [data_normalized[i:i+3] for i in range(0, len(data_normalized) - 3)]
train_y = [data_normalized[i+3] for i in range(len(data_normalized) - 3)]
random.seed(23)

# Ініціалізація випадкових ваг і порогових значень (biases) для нейронної мережі
w_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(3)] # Ваги між входами та прихованими шарами
w_output_hidden = [random.uniform(-0.5, 0.5) for _ in range(3)] # Ваги між прихованими шарами та виходом
b_hidden = [random.uniform(-0.1, 0.1) for _ in range(3)] # Порогові значення для прихованих шарів
b_output = random.uniform(-0.1, 0.1) # Порогові значення для виходу

learning_rate = 0.1  # Швидкість навчання
max_iterations = 2_500_000  # Максимальна кількість ітерацій для тренування
tolerance = 0.0001 # Допустима похибка для завершення навчання

# Основний цикл тренування нейронної мережі
for epoch in range(max_iterations):
    total_error = 0
    for i in range(len(train_x)):
        inputs, expected = train_x[i], train_y[i]

        # Обчислення активацій прихованих нейронів
        hidden_activations = [sigmoid(sum(inputs[k] * w_input_hidden[j][k] for k in range(3)) + b_hidden[j]) for j in range(3)]

        # Обчислення вихідного результату
        output = sigmoid(sum(hidden_activations[j] * w_output_hidden[j] for j in range(3)) + b_output)

        # Помилка між передбаченням і очікуваним результатом
        error = expected - output
        total_error += error ** 2

        # Обчислення змінних для back propagation
        delta_output = error * sigmoid_derivative(output)
        delta_hidden = [delta_output * w_output_hidden[j] * sigmoid_derivative(hidden_activations[j]) for j in range(3)]

        # Оновлення ваг та порогових значень
        for j in range(3):
            w_output_hidden[j] += learning_rate * delta_output * hidden_activations[j]
            for k in range(3):
                w_input_hidden[j][k] += learning_rate * delta_hidden[j] * inputs[k]
            b_hidden[j] += learning_rate * delta_hidden[j]
        b_output += learning_rate * delta_output

    # Виведення MSE (середньоквадратичної помилки) кожні 10,000 ітерацій
    if epoch % 10_000 == 0:
        print(f"Epoch {epoch}, MSE: {total_error / len(train_x):.6f}")

    # Завершення тренування, якщо досягнута достатня точність
    if total_error / len(train_x) < tolerance:
        print("Finished")
        break

# Перевірка моделі на тестових даних
total_test_error = 0

for i, test_input in enumerate(test_data):
    # Обчислення активацій прихованих нейронів для тестових даних
    hidden_activations = [sigmoid(sum(test_input[k] * w_input_hidden[j][k] for k in range(3)) + b_hidden[j]) for j in range(3)]

    # Обчислення тестового передбачення
    test_prediction = sigmoid(sum(hidden_activations[j] * w_output_hidden[j] for j in range(3)) + b_output)

    # Денормалізація результату
    denormalized_prediction = denormalize(test_prediction, min_value, max_value)
    print(f"Predicted x{i + 13}: {denormalized_prediction:.4f}")
    # Обчислення помилки для тестового передбачення
    if i < len(expected_result):
        total_test_error += (denormalized_prediction - denormalize(expected_result[i], min_value, max_value)) ** 2

# Виведення середньої помилки на тестових даних
print(f"Test error: {total_test_error / len(test_data):.6f}")