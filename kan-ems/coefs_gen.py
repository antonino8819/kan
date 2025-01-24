import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Passo 1: Generare il vettore base_weights
k_elements = 10000
base_weights = np.random.uniform(0, 1, k_elements)

# Numero di iterazioni T
T = 100

# Liste per salvare le distanze medie punto-punto per ogni step nel primo caso (trasformazioni)
distances_transform = []

# Liste per salvare le distanze medie punto-punto per ogni step nel secondo caso (random vettori)
distances_random = []

# Passo 2: Creare le classi basate su intervalli di valori
m_classes = 300
interval_size = 1 / m_classes
classes = []
class_indices = []

for i in range(m_classes):
    lower_bound = i * interval_size
    upper_bound = (i + 1) * interval_size
    class_elements = base_weights[(base_weights >= lower_bound) & (base_weights < upper_bound)]
    indices = np.where((base_weights >= lower_bound) & (base_weights < upper_bound))[0]
    classes.append(class_elements)
    class_indices.append(indices)

# Ripetere il processo T volte per le trasformazioni
for t in range(T):
    # Passo 3: Creare un vettore individual con valori casuali uniformi
    individual = np.random.uniform(0, 1, m_classes)
    # individual = np.full(m_classes, 0.5)

    # Passo 4: Trasformare i valori delle classi in base al rispettivo valore di alpha in individual
    transformed_classes = []
    transformed_indices = []

    for alpha, class_elements, indices in zip(individual, classes, class_indices):
        if alpha < 0.5:
            transformed = alpha * class_elements
        else:
            transformed = 2 * (1 - alpha) * class_elements + 2 * alpha - 1
        transformed_classes.append(transformed)
        transformed_indices.append(indices)

    # Calcolare la distanza media punto-punto rispetto a base_weights
    all_transformed_values = np.concatenate(transformed_classes)
    all_transformed_indices = np.concatenate(transformed_indices)
    distance = np.mean(np.abs(all_transformed_values - base_weights[all_transformed_indices]))

    distances_transform.append(distance)

    # Plot delle transformed_classes per questa iterazione
    '''
    plt.figure(figsize=(10, 6))
    for t_class, indices in zip(transformed_classes, transformed_indices):
        plt.scatter(indices, t_class, s=10, color='red')

    plt.title(f'Transformed Classes (Step {t + 1})')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    '''

# Ripetere il processo T volte per i vettori random
for t in range(T):
    # Generare un vettore random simile a base_weights
    random_vector = np.random.uniform(0, 1, k_elements)

    # Calcolare la distanza media punto-punto rispetto a base_weights
    distance = np.mean(np.abs(random_vector - base_weights))

    distances_random.append(distance)

    # Plot del vettore random per questa iterazione
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(random_vector)), random_vector, s=10, color='green')
    plt.title(f'Random Vector (Step {t + 1})')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    '''

mean_distance_transform = np.mean(distances_transform)
mean_distance_random = np.mean(distances_random)

# Calcolare il range delle distanze casuali
random_max = max(distances_random)
random_min = min(distances_random)

# Identificare i punti di distances_transform che rientrano nel range o che sono superiori al massimo
highlight_points = [(i + 1, val) for i, val in enumerate(distances_transform) if random_min <= val <= random_max or val > random_max]
highlight_count = len(highlight_points)

# Passo 5: Grafico delle distanze medie punto-punto per entrambi i casi
plt.figure(figsize=(10, 6))

# Grafico delle distanze
plt.plot(range(1, T + 1), distances_transform, label='Mean Point-to-Point Distance (Transform)', marker='o', color='blue')
plt.plot(range(1, T + 1), distances_random, label='Mean Point-to-Point Distance (Random)', marker='o', color='green')

# Tracciare il rettangolo demitrasparente
rect = patches.Rectangle((0.5, random_min), T, random_max - random_min, color='green', alpha=0.2, label='Random Range')
plt.gca().add_patch(rect)

# Evidenziare i punti
if highlight_points:
    highlight_x, highlight_y = zip(*highlight_points)
    plt.scatter(highlight_x, highlight_y, color='red', label='Highlighted Points')

# Aggiungere le linee orizzontali per le medie
plt.axhline(y=mean_distance_transform, color='blue', linestyle='--', label=f'Mean Transform = {mean_distance_transform:.2f}')
plt.axhline(y=mean_distance_random, color='green', linestyle='--', label=f'Mean Random = {mean_distance_random:.2f}')

# Personalizzare il grafico
plt.title('Mean Point-to-Point Distance Across Steps')
plt.xlabel('Step')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)

# Stampare il numero di punti evidenziati
print(f"Numero di punti evidenziati: {highlight_count}")

# Mostrare il grafico
plt.show()


# Generare un individual random di riferimento
reference_individual = np.random.uniform(0, 1, m_classes)

# Generare nuovi individuals con modifiche graduali
modified_individuals = [
    reference_individual.copy(),  # Nessun cambiamento
    np.array([0.5 if i == np.random.randint(m_classes) else v for i, v in enumerate(reference_individual)]),  # Cambia un solo elemento
    np.array([0.5 if i in np.random.choice(m_classes, m_classes // 4, replace=False) else v for i, v in enumerate(reference_individual)]),  # Cambia un quarto
    np.array([0.5 if i in np.random.choice(m_classes, m_classes // 2, replace=False) else v for i, v in enumerate(reference_individual)]),  # Cambia metà
    np.random.uniform(0, 1, m_classes)  # Cambiano tutti
]

# Calcolare le distanze medie rispetto al reference_individual
distances_from_reference = [
    np.mean(np.abs(modified - reference_individual)) for modified in modified_individuals
]

# Grafico delle distanze
plt.figure(figsize=(10, 6))
plt.plot(range(len(distances_from_reference)), distances_from_reference, marker='o', color='purple', label='Distance from Reference')
plt.xticks(range(len(distances_from_reference)), ['No Change', 'One Change', 'Quarter Changed', 'Half Changed', 'All Changed'])
plt.title('Mean Distance from Reference Individual')
plt.xlabel('Modification Type')
plt.ylabel('Mean Distance')
plt.legend()
plt.grid(True)
plt.show()
