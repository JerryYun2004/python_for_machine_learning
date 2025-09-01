import matplotlib.pyplot as plt

# Data
models = ['MLP', 'CNN_0', 'CNN_1', 'CNN_2', 'CNN_3']
accuracy = [42.5, 4.76, 21.11, 19.21, 55.17]
loss = [None, 3.6894, 2.8027, 2.8372, 1.5032]

# Plot Accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(models, accuracy, color='orange')
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 60)

# Plot Loss (excluding MLP which has no loss value)
plt.subplot(1, 2, 2)
loss_models = [m for m, l in zip(models, loss) if l is not None]
loss_values = [l for l in loss if l is not None]
plt.bar(loss_models, loss_values, color='skyblue')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.ylim(0, 4)

plt.show()