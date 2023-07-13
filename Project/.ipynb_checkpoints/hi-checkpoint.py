import numpy as np

class HopfieldNetwork:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.weights = np.zeros((dimensions, dimensions))

    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.flatten()
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, iterations=5):
        pattern = pattern.flatten()
        for _ in range(iterations):
            for i in range(self.dimensions):
                raw_output = np.dot(self.weights[i], pattern)
                if raw_output > 0:
                    pattern[i] = 1
                elif raw_output < 0:
                    pattern[i] = -1
        return pattern.reshape((int(np.sqrt(self.dimensions)), int(np.sqrt(self.dimensions))))


from tensorflow.keras.datasets import mnist

# Lade den MNIST-Datensatz herunter
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisiere die Daten auf -1 bis 1
x_train = x_train / 127.5 - 1
x_test = x_test / 127.5 - 1

# Verflache die Bilder
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))


# Erstelle ein Hopfield-Netzwerk
network = HopfieldNetwork(28*28)

# Trainiere das Netzwerk mit den Trainingsdaten
network.train(x_train)


# Wähle ein zufälliges Bild aus dem Testset
input_image = np.copy(x_test[np.random.choice(len(x_test))])

# Füge Rauschen hinzu, indem du zufällig einige Pixel umdrehst
# Hierbei wird nur mit einer Wahrscheinlichkeit von 10% ein Pixel umgedreht
noise_indices = np.random.choice([True, False], size=input_image.shape, p=[0.1, 0.9])
input_image[noise_indices] = -input_image[noise_indices]


# Versuche, das ursprüngliche Bild zurückzurufen
output_image = network.recall(input_image)


import matplotlib.pyplot as plt

# Forme die Bilder in ihre ursprüngliche Form von 28x28 um
input_image = input_image.reshape((28, 28))
output_image = output_image.reshape((28, 28))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Eingabebild')
plt.imshow(input_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Ausgabebild')
plt.imshow(output_image, cmap='gray')

plt.show()