import matplotlib.pyplot as plt

# Updated accuracy values
models = ['CNN_0', 'CNN_1', 'CNN_2', 'CNN_3', 'CNN_Final']
accuracy = [4.76, 21.11, 19.21, 55.17, 72.4]

# Create the bar graph with increased font size
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color='orange')
plt.xlabel('Model', fontsize=25)
plt.ylabel('Accuracy (%)', fontsize=25)
plt.title('Model Accuracy Comparison', fontsize=30)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 100)

# Annotate bars with accuracy values
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=14)

plt.tight_layout()
plt.show()
