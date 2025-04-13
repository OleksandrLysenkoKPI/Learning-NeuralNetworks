import numpy as np
import random
import matplotlib.pyplot as plt

def func(x):
    return -x ** np.cos(5 * x)

def select_parents(population, fitness, num_parents):
    probabilities = np.exp(fitness - np.max(fitness))
    probabilities /= probabilities.sum()
    selected = np.random.choice(population, size=num_parents, p=probabilities, replace=False)
    return selected.tolist()

def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        p1, p2 = random.sample(parents, 2)
        child = (p1 + p2) / 2
        offspring.append(child)
    return offspring


def mutate(offspring, mutation_rate, bounds):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] += random.uniform(-0.1, 0.1)
            offspring[i] = np.clip(offspring[i], bounds[0], bounds[1])
    return offspring


def genetic_algorithm(bounds, maximize=True, pop_size=1200, generations=50,
                      mutation_rate=0.2, elitism=0.1, patience=5):

    population = [random.uniform(bounds[0], bounds[1]) for _ in range(pop_size)]
    best_solution = None
    if maximize:
        best_value = -np.inf
    else:
        best_value = np.inf

    elite_size = int(pop_size * elitism)
    no_improve_count = 0

    for generation in range(generations):
        fitness = np.array([func(x) for x in population])

        if not maximize:
            sorted_indices = np.argsort(fitness)
        else:
            sorted_indices = np.argsort(fitness)[::-1]

        elites = [population[i] for i in sorted_indices[:elite_size]]

        if (maximize and fitness[sorted_indices[0]] > best_value) or (not maximize and fitness[sorted_indices[0]] < best_value):
            best_value = fitness[sorted_indices[0]]
            best_solution = population[sorted_indices[0]]
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count > patience:
            print(f'Алгоритм зупинився на поколінні {generation + 1} через відсутність покращень.')
            break

        parents = select_parents(population, fitness, num_parents=pop_size // 2)
        offspring = crossover(parents, offspring_size=pop_size - len(elites))
        offspring = mutate(offspring, mutation_rate, bounds)
        population = elites + offspring

        print(f'Покоління {generation + 1}: найкраще значення = {best_value:.6f}')

    return best_solution, best_value


bounds = [0, 10]

print("Пошук максимумів:")
max_x_gen, max_y_gen = genetic_algorithm(bounds, maximize=True)
print("\nПошук мінімумів:")
min_x_gen, min_y_gen = genetic_algorithm(bounds, maximize=False)

x_val = np.linspace(bounds[0], bounds[1], 1000)
y_val = func(x_val)

min_index = np.argmin(y_val)
max_index = np.argmax(y_val)

plt.figure(figsize=(10, 6))
plt.plot(x_val, y_val, label="Y(x)", linewidth=2)

# Істинний мінімум і максимум (аналітичні з np.argmin/argmax)
plt.plot(x_val[min_index], y_val[min_index], 'r^', label="Істинний мінімум", markersize=10)
plt.plot(x_val[max_index], y_val[max_index], 'g^', label="Істинний максимум", markersize=10)

# Генетичні розв'язки
plt.scatter([max_x_gen], [max_y_gen], color='black', label='Максимум (GA)', s=60, marker='o')
plt.scatter([min_x_gen], [min_y_gen], color='violet', label='Мінімум (GA)', s=60, marker='o')

# Підписи
plt.text(x_val[min_index], y_val[min_index] - 2, f"True Min\n({x_val[min_index]:.2f}, {y_val[min_index]:.2f})",
         ha='center', fontsize=8, color='red')
plt.text(x_val[max_index], y_val[max_index] + 2, f"True Max\n({x_val[max_index]:.2f}, {y_val[max_index]:.2f})",
         ha='center', fontsize=8, color='green')
plt.text(max_x_gen, max_y_gen + 2, f"GA Max\n({max_x_gen:.2f}, {max_y_gen:.2f})",
         ha='center', fontsize=8, color='black')
plt.text(min_x_gen, min_y_gen - 2, f"GA Min\n({min_x_gen:.2f}, {min_y_gen:.2f})",
         ha='center', fontsize=8, color='purple')

plt.legend()
plt.title("Пошук екстремумів функції Y(x)")
plt.xlabel("x")
plt.ylabel("Y(x)")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f'Максимум: Y({max_x_gen:.4f}) = {max_y_gen:.4f}')
print(f'Мінімум: Y({min_x_gen:.4f}) = {min_y_gen:.4f}')
print(f'Справжній максимум: Y({x_val[max_index]:.2f}) = {y_val[max_index]:.2f}')
print(f'Справжній мінімум: Y({x_val[min_index]:.2f}) = {y_val[min_index]:.2f}')