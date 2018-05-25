import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as ks


class CONFIG:
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    CONTENT_LAYER = 'block4_conv2'
    INPUT_LAYER = 'input_1'


def generate_noise_image(content_image, noise_ratio=CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20,
                                    (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype(
        'float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image


def get_image(image_path, target_size=(CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)):
    img = ks.preprocessing.image.load_img(image_path, target_size=target_size)
    img = ks.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = ks.applications.vgg19.preprocess_input(img)
    return img


def compute_content_cost(content_image_activations, generate_image_activations):
    return tf.divide(tf.reduce_mean(tf.square(tf.subtract(content_image_activations, generate_image_activations))), 4.0)


def main():
    vgg19 = ks.applications.vgg19.VGG19(weights='imagenet')
    print(tf.trainable_variables())
    # print(vgg19.summary())
    with tf.variable_scope('content_model'):

        content_model = ks.models.Model(inputs=vgg19.input, outputs=vgg19.get_layer(CONFIG.CONTENT_LAYER).output)
        print(tf.trainable_variables(scope='content_model'))

    content_model.save_weights('./weight/content.hdf5')
    content_img = get_image('./images/Coffee-Mug.jpg')
    generate_img = generate_noise_image(content_img)

    content_activations = content_model.predict(content_img)

    g_img = tf.placeholder(dtype=tf.float32, shape=(1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS))
    generate_activations = content_model(g_img)
    content_cost = compute_content_cost(content_activations, generate_activations)

    # print(list(set(tf.trainable_variables()) - set(content_model.trainable_variables)))
    # train_step = tf.train.AdamOptimizer(2.0).minimize(content_cost, var_list=tf.trainable_variables() - content_model.trainable_variables)

    # print(content_model.get_weights()[0][0][0][0])
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     content_model.load_weights('./weight/content.hdf5')
    #     print(generate_img.shape)
    #     for iter_num in range(5):
    #         print(sess.run(content_cost, feed_dict={g_img: generate_img}))
    #         sess.run(train_step, feed_dict={g_img: generate_img})
            # print(content_model.get_weights()[0][0][0][0][0])
        # print(content_model.get_weights()[0][0][0][0])

    # content_image_activations = sess.run(content_layer)
    # generate_image_activations = content_layer
    # content_cost = compute_content_cost(content_image_activations, generate_image_activations)

    print('it is done')


main()