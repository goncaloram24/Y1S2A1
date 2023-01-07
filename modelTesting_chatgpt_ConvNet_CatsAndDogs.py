import tensorflow as tf
import cv2
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("chatgpt_ConvNet_CatsAndDogs_models/model_epoch10.h5")


# Load an image
cat = cv2.imread("chatgpt_ConvNet_CatsAndDogs_pics/cat.jpg")
dog = cv2.imread("chatgpt_ConvNet_CatsAndDogs_pics/dog.jpg")

cat_resized = cv2.resize(cat, (32, 32), interpolation=cv2.INTER_LINEAR)
dog_resized = cv2.resize(dog, (32, 32), interpolation=cv2.INTER_LINEAR)

cat_resized = cat_resized[None,:,:,:]
dog_resized = dog_resized[None,:,:,:]

pred1 = round(model.predict(cat_resized).item(0))
pred2 = round(model.predict(dog_resized).item(0))

print(str(pred1))
print(str(pred2))

#---------------------------------
# Display the image
cv2.imshow('Cat', cat)
cv2.imshow('Dog', dog)

# Wait until a key is pressed
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
