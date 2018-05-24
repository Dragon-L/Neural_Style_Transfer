import tensorflow as tf
import tensorflow.contrib.keras as ks

img = ks.preprocessing.image.load_img('./images/car.jpg', target_size=(224, 224))
img = ks.preprocessing.image.img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

vgg19 =ks.applications.vgg19.VGG19(weights='imagenet')
yhat = vgg19.predict(img)
label = ks.applications.vgg19.decode_predictions(yhat)
print(label[0][0])

print('it is done')