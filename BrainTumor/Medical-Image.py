import matplotlib.pyplot as plt
from utils import match_images_with_annotations as match
from utils import create_df,get_df
from model import cnn

#file-path
train_path = "/home/hamidreza/Downloads/Others/archive/train"
test_path = "/home/hamidreza/Downloads/Others/archive/test"
validation_path = "/home/hamidreza/Downloads/Others/archive/valid"

#annotation-path
train_annotations_path = "/home/hamidreza/Downloads/Others/archive/train/_annotations.coco.json"
test_annotations_path = "/home/hamidreza/Downloads/Others/archive/test/_annotations.coco.json"
validation_annotations_path = "/home/hamidreza/Downloads/Others/archive/valid/_annotations.coco.json"

#json-view
#print(jskeys(validation_annotations_path))

#match-annotation
match_train_data = match(train_path,train_annotations_path)
match_validation_data = match(validation_path,validation_annotations_path)
match_test_data = match(test_path,test_annotations_path)
#display-annotation
#display(matchdata,4)

#image-details
#train_images = idg(rescale=1./255)

#split-image-label
train_df = create_df(match_train_data)
train_data = get_df(train_df)
valid_df = create_df(match_validation_data)
valid_data = get_df(valid_df)
test_df = create_df(match_test_data)
test_data = get_df(test_df)

#train-model
model = cnn()
history = model.fit(train_data,batch_size=32,epochs=15,validation_data=(valid_data),shuffle=True)

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot accuracy on the first subplot (ax1)
ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_xticks([0, 5, 10, 15])
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot loss on the second subplot (ax2)
ax2.plot(history.history['loss'], label='Training Loss', color='orange')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
ax2.set_title('Loss')
ax2.set_xlabel('Epochs')
ax2.set_xticks([0, 5, 10, 15])
ax2.set_ylabel('Loss')
ax2.legend()

# Evaluate the model on test data
loss, acc = model.evaluate(test_data, batch_size=32)
print("Test Accuracy:", acc)
print("Test Loss:", loss)

# Plot test accuracy and test loss on respective subplots
epochs = range(1, len(history.history['accuracy']) + 1)  # Generate epochs from 1 to the number of epochs
ax1.plot(epochs, [acc] * len(epochs), label="Test Accuracy", color="black", linestyle="dotted")
ax2.plot(epochs, [loss] * len(epochs), label="Test Loss", color="black", linestyle="dotted")

# Show the plot
plt.tight_layout()
plt.show()
