# Get a batch of images and labels from the generator
batch = next(train_data)

# Extract the images and labels from the batch
images, labels = batch

# Display a subset of images
num_images = 5  # Number of images to display
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
for i in range(num_images):
    axes[i].imshow(images[i])
    axes[i].set_title(f"Label: {labels[i]}")
    axes[i].axis('off')

plt.show()