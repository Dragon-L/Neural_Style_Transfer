import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as ks
from scipy.misc import imread, imresize, imsave, imshow
import matplotlib.pyplot as plt

from utility import show_images, reshape_and_normalize_image, generate_noise_image


class CONFIG:
    CONTENT_IMAGE_PATH = './images/house.jpg'
    STYLE_IMAGE_PATH = './images/style_image.jpg'
    GENERATED_IMAGE_PATH = './images/generate_image.jpg'
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.0
    CONTENT_LAYER = 'block4_conv2'
    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]
    INPUT_LAYER = 'input_1'
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


def main():
    first_time = True

    content_img = imread(CONFIG.CONTENT_IMAGE_PATH)
    style_img = imread(CONFIG.STYLE_IMAGE_PATH)
    content_img = reshape_and_normalize_image(content_img, CONFIG.MEANS, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))
    style_img = reshape_and_normalize_image(style_img, CONFIG.MEANS, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))

    if first_time:
        generate_img = generate_noise_image(content_img, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS, CONFIG.NOISE_RATIO)
    else:
        generate_img = imread(CONFIG.GENERATED_IMAGE_PATH)
        generate_img = np.expand_dims(generate_img, axis=0)

    save_image(generate_img, 'generate_img_0')
    input_img = tf.get_variable(name='input_img', dtype=tf.float32, shape=(1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS))
    vgg19 = ks.applications.vgg19.VGG19(weights='imagenet', input_tensor=tf.convert_to_tensor(input_img))
    weight = vgg19.get_weights()

    content_cost = compute_content_cost(content_img, input_img, vgg19, weight, CONFIG.CONTENT_LAYER)
    style_cost = compute_style_cost(input_img, style_img, vgg19, weight, CONFIG.STYLE_LAYERS)
    total_cost = 10 * content_cost + 40 * style_cost

    train_step = tf.train.AdamOptimizer(20.0).minimize(total_cost, var_list=input_img)

    summary = tf.summary.scalar('total_cost', total_cost)
    writer = tf.summary.FileWriter('./tensorboard/train', tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vgg19.set_weights(weights=weight)
        sess.run(tf.assign(input_img, generate_img))
        for iteration_num in range(100):
            summary_cost ,cost, _ = sess.run([summary, total_cost, train_step])
            print('iteration ' + str(iteration_num) + ' : ' + str(cost))
            writer.add_summary(summary_cost, iteration_num)
            if iteration_num % 10 == 0:
                output_image = sess.run(input_img)
                save_image(output_image, 'generated_image_' + str(iteration_num))
        output_image = sess.run(input_img)

    writer.close()
    save_image(output_image, 'generate_image')
    # plt.imshow(np.squeeze(output_image))
    # plt.show()
    print('it is done')


def save_image(output_image, image_name):
    # output_image = output_image + CONFIG.MEANS
    output_image = np.squeeze(output_image)
    output_image = np.clip(output_image, 0, 255).astype('uint8')
    imsave('./images/' + image_name + '.jpg', output_image)


def compute_style_cost(generated_img, style_img, model, pretrained_weight, output_layers):
    style_cost = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.set_weights(weights=pretrained_weight)
        sess.run(tf.assign(generated_img, style_img))
        for (layer, coeff) in output_layers:
            style_output = model.get_layer(layer).output
            style_activations = sess.run(style_output)
            layer_style_cost = compute_style_layer_cost(style_activations, style_output)
            style_cost += coeff * layer_style_cost
    return style_cost


def compute_style_layer_cost(style_image_activations, generate_image_activations):
    m, n_h, n_w, n_c = style_image_activations.shape
    s_matrix = tf.reshape(style_image_activations, (-1, n_c))
    g_matrix = tf.reshape(generate_image_activations, (-1, n_c))
    s_mul = tf.matmul(tf.transpose(s_matrix), s_matrix)
    g_mul = tf.matmul(tf.transpose(g_matrix), g_matrix)
    style_cost = tf.divide(tf.reduce_sum(tf.square(tf.subtract(s_mul, g_mul))), tf.square(2.0 * n_h * n_w * n_c))
    return style_cost


def compute_content_cost(content_img, generated_img, model, pretrained_weight, output_layer):
    output = model.get_layer(output_layer).output
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.set_weights(weights=pretrained_weight)
        sess.run(tf.assign(generated_img, content_img))
        content_activations = sess.run(output)
    content_cost = tf.divide(tf.reduce_mean(tf.square(tf.subtract(content_activations, output))), 4.0)
    return content_cost


main()





