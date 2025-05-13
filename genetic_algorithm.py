import numpy as np
import matplotlib.pyplot as plt

# Цільова функція: f(x) = -(x/256) + 4x + 200
def objective_function(x):
    return -(x / 256) + 4 * x + 200

# Ініціалізація популяції
def initialize_population(size, bounds):
    return np.random.randint(bounds[0], bounds[1] + 1, size=size)

# Оцінка пристосованості
def evaluate_population(population):
    return np.array([objective_function(ind) for ind in population])

# Селекція (турнірний відбір)
def select_parents(population, fitness):
    parents = []
    for _ in range(2):
        candidates = np.random.choice(len(population), size=2, replace=False)
        parents.append(population[candidates[np.argmax(fitness[candidates])]])
    return parents

# Схрещування (одна точка перетину)
def crossover(parents, pc):
    if np.random.rand() < pc:
        point = np.random.randint(1, 8)  # Точка перетину для 8-бітного представлення
        mask = (1 << point) - 1
        child1 = (parents[0] & mask) | (parents[1] & ~mask)
        child2 = (parents[1] & mask) | (parents[0] & ~mask)
        return child1, child2
    return parents[0], parents[1]

# Мутація (фліп бітів)
def mutate(individual, pm):
    for i in range(8):  # 8-бітне представлення
        if np.random.rand() < pm:
            individual ^= 1 << i  # Інвертуємо біт
    return individual

# Основна функція генетичного алгоритму
def genetic_algorithm(bounds, n, pc, pm, generations=100):
    population = initialize_population(n, bounds)
    best_solution = None
    best_fitness = -np.inf

    for generation in range(generations):
        fitness = evaluate_population(population)

        if np.max(fitness) > best_fitness:
            best_fitness = np.max(fitness)
            best_solution = population[np.argmax(fitness)]

        new_population = []
        for _ in range(n // 2):
            parents = select_parents(population, fitness)
            child1, child2 = crossover(parents, pc)
            new_population.append(mutate(child1, pm))
            new_population.append(mutate(child2, pm))

        population = np.array(new_population)

    return best_solution, best_fitness

# Параметри задачі
bounds = (0, 255)
n = 36  # Розмір популяції
pc = 0.62  # Ймовірність схрещування
pm = 0.005  # Ймовірність мутації

# Виконання алгоритму
best_solution, best_fitness = genetic_algorithm(bounds, n, pc, pm)
print(f"Найкраще рішення: x = {best_solution}")
print(f"Максимальне значення цільової функції: f(x) = {best_fitness}")

# Візуалізація функції
x = np.linspace(bounds[0], bounds[1], 1000)
y = objective_function(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = -(x/256) + 4x + 200', color='blue')
plt.scatter(best_solution, best_fitness, color='red', label=f'Best Solution: x = {best_solution}, f(x) = {best_fitness}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Максимізація функції за допомогою ГА')
plt.legend()
plt.grid()
plt.show()
# Ось ТАКА змінА

