import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as ks
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt


class CONFIG:
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    CONTENT_LAYER = 'block4_conv2'
    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]
    INPUT_LAYER = 'input_1'
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def reshape_and_normalize_image(image):
    image = np.reshape(image, ((1,) + image.shape))
    # image = np.expand_dims(image)

    image = image - CONFIG.MEANS

    return image


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


def compute_content_cost(content_image_activations, generate_image_activations):
    return tf.divide(tf.reduce_mean(tf.square(tf.subtract(content_image_activations, generate_image_activations))), 4.0)


def compute_style_layer_cost(style_image_activations, generate_image_activations):
    m, n_h, n_w, n_c = style_image_activations.shape
    s_matrix = tf.reshape(style_image_activations, (-1, n_c))
    g_matrix = tf.reshape(generate_image_activations, (-1, n_c))
    s_mul = tf.matmul(tf.transpose(s_matrix), s_matrix)
    g_mul = tf.matmul(tf.transpose(g_matrix), g_matrix)
    style_cost = tf.divide(tf.reduce_sum(tf.square(tf.subtract(s_mul, g_mul))), tf.square(2.0 * n_h * n_w * n_c))
    return style_cost


def main():
    first_time = False
    images = []
    content_img = imread('./images/dou.jpg')
    style_img = imread('./images/style_image.jpg')
    content_img = imresize(content_img, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))
    style_img = imresize(style_img, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))
    content_img = reshape_and_normalize_image(content_img)
    style_img = reshape_and_normalize_image(style_img)

    if first_time:
        generate_img = generate_noise_image(content_img)
    else:
        generate_img = imread('./images/generate_image.jpg')
        generate_img = np.expand_dims(generate_img, axis=0)

    # images.append(np.squeeze(generate_img))

    input_img = tf.get_variable(name='input_img', dtype=tf.float32, shape=(1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS))
    vgg19 = ks.applications.vgg19.VGG19(weights='imagenet', input_tensor=tf.convert_to_tensor(input_img))
    weight = vgg19.get_weights()

    content_output = vgg19.get_layer(CONFIG.CONTENT_LAYER).output
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vgg19.set_weights(weights=weight)
        sess.run(tf.assign(input_img, content_img))
        content_activations = sess.run(content_output)
    content_cost = compute_content_cost(content_activations, content_output)

    style_cost = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vgg19.set_weights(weights=weight)
        sess.run(tf.assign(input_img, style_img))
        for (layer, coeff) in CONFIG.STYLE_LAYERS:
            style_output = vgg19.get_layer(layer).output
            style_activations = sess.run(style_output)
            layer_style_cost = compute_style_layer_cost(style_activations, style_output)
            style_cost += coeff * layer_style_cost

    total_cost = 10 * content_cost + 40 * style_cost
    train_step = tf.train.AdamOptimizer().minimize(total_cost, var_list=input_img)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vgg19.set_weights(weights=weight)
        sess.run(tf.assign(input_img, generate_img))
        for iteration_num in range(2000):
            cost, _ = sess.run([content_cost, train_step])
            print('iteration ' + str(iteration_num) + ' : ' + str(cost))
            # if iteration_num % 50 == 0:
                # output_image = sess.run(input_img)
                # temp = np.squeeze(output_image)
                # imsave('./images/generate_image_' + str(iteration_num) + '.jpg', temp)
        output_image = sess.run(input_img)

    output_image = np.squeeze(output_image)
    imsave('./images/generate_image.jpg', output_image)
    # images.append(output_image)
    # show_images(images)
    # plt.imshow(output_image)
    # plt.show()
    print('it is done')


main()