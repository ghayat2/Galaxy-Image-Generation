import tensorflow as tf
import layers
import sys
import time

global_seed = 5


class MCGAN:
    def __init__(self):
        return

    def generator_model(self, noise, feats, training, reuse=False):  # construct the graph of the generator
        a = time.time()

        with tf.variable_scope("generator",
                               reuse=reuse):  # define variable scope to easily retrieve vars of the generator

            gen_inp = tf.concat([noise, feats], -1)
            with tf.name_scope("preprocess_inp"):
                dense1 = layers.dense_layer(gen_inp, units=4 * 4 * 1024, use_bias=False)
                bn1 = layers.batch_norm_layer_mcgan(dense1, training, 0.8)
                relu1 = layers.relu_layer(bn1)
                reshaped = tf.reshape(relu1, shape=[-1, 1024, 4, 4])  # shape=(batch_size, 1024, 4, 4)

            deconv1 = layers.deconv_block_mcgan(reshaped, training, momentum=0.8, out_channels=512, filter_size=(4, 4),
                                                strides=(2, 2), padding="same",
                                                use_bias=True)  # shape=(batch_size, 512, 8, 8)
            deconv2 = layers.deconv_block_mcgan(deconv1, training, momentum=0.8, out_channels=256, filter_size=(4, 4), strides=(2, 2),
                                                padding="same", use_bias=True)  # shape=(batch_size, 256, 16, 16)
            deconv3 = layers.deconv_block_mcgan(deconv2, training, momentum=0.8, out_channels=128, filter_size=(4, 4), strides=(2, 2),
                                                padding="same", use_bias=True)  # shape=(batch_size, 128, 32, 32)
            deconv4 = layers.deconv_layer(deconv3, out_channels=1, filter_size=(4, 4), strides=(2, 2), padding="same",
                                          use_bias=True)  # shape=(batch_size, 1, 64, 64)

            gen_out = layers.tanh_layer(deconv4)
        print("Built Generator model in {} s".format(time.time() - a))
        list_ops = {"dense1": dense1, "bn1":bn1, "relu1": relu1, "reshaped": reshaped, "deconv1": deconv1, "deconv2": deconv2,
                    "deconv3": deconv3, "deconv4": deconv4,
                    "gen_out": gen_out}  # list of operations, can be used to run the graph up to a certain op
        # i,e get the subgraph
        return gen_out, list_ops

    def discriminator_model(self, inp, feats, training, reuse=False, resize=False, minibatch=False):  # construct the graph of the discriminator
        a = time.time()
        with tf.variable_scope("discriminator",
                               reuse=reuse):  # define variable scope to easily retrieve vars of the discriminator

            if resize:
                inp = layers.max_pool_layer(inp, pool_size=(2, 2), strides=(2, 2), padding=(12, 12))
                inp = layers.max_pool_layer(inp, pool_size=(2, 2), strides=(2, 2))
                inp = layers.max_pool_layer(inp, pool_size=(2, 2), strides=(2, 2))
                inp = layers.max_pool_layer(inp, pool_size=(2, 2), strides=(2, 2))

            conv1 = layers.conv_block_mcgan(inp, training, momentum=0.8, out_channels=128, filter_size=(4, 4), strides=(2, 2),
                                            padding="same", use_bias=True, batch_norm=False,
                                            alpha=0.3)  # shape=(batch_size, 128, 32, 32)
            conv2 = layers.conv_block_mcgan(conv1, training, momentum=0.8, out_channels=256, filter_size=(4, 4), strides=(2, 2),
                                            padding="same", use_bias=True,
                                            alpha=0.3)  # shape=(batch_size, 256, 16, 16)
            conv3 = layers.conv_block_mcgan(conv2, training, momentum=0.8, out_channels=512, filter_size=(4, 4), strides=(2, 2),
                                            padding="same", use_bias=True, alpha=0.3)  # shape=(batch_size, 512, 8, 8)
            conv4 = layers.conv_block_mcgan(conv3, training, momentum=0.8, out_channels=1024, filter_size=(4, 4), strides=(2, 2),
                                            padding="same", use_bias=True, alpha=0.3)  # shape=(batch_size, 1024, 4, 4)
            flat = tf.reshape(conv4, [-1, 1024 * 4 * 4])

            if(minibatch):
                minibatched = layers.minibatch(flat, num_kernels=5, kernel_dim=3)
                dense1 = layers.dense_layer(minibatched, 128, use_bias=True)
            else:
                dense1 = layers.dense_layer(flat, 128, use_bias=True)
            
            drop1 = layers.dropout_layer(dense1, training, dropout_rate=0.3)
            LRU1 = layers.leaky_relu_layer(drop1, alpha=0.3)

            dense2 = layers.dense_layer(feats, units=3, use_bias=True)
            relu2 = layers.relu_layer(dense2)
            bn2 = layers.batch_norm_layer_mcgan(relu2, training, 0.8)
            dense3 = layers.dense_layer(bn2, units=1)

            merged = tf.concat([LRU1, dense3], axis=-1)

            logits = layers.dense_layer(merged, units=2, use_bias=True)
            out = logits
        print("Built Discriminator model in {} s".format(time.time() - a))

        list_ops = {"inp": inp, "conv1": conv1, "conv2": conv2, "conv3": conv3, "conv4": conv4, "flat": flat,
                    "dense1":dense1, "drop1":drop1, "LRU1":LRU1, "dense2":dense2, "relu2":relu2, "bn2":bn2,
                    "dense3":dense3, "logits": logits, "out": out}

        return out, list_ops

    def generator_vars(self):
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        return gen_vars

    def discriminator_vars(self):
        discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return discr_vars

    def generator_loss(self, fake_out, labels, label_smoothing=False):
        with tf.name_scope("generator_loss"):
            if(label_smoothing):
                smoothed_labels = tf.one_hot(labels, depth=2)
                loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_labels, logits=fake_out))
            else:
                loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fake_out))
        return loss_gen

    def discriminator_loss(self, fake_out, real_out, fake_labels, real_labels, label_smoothing=False):
        with tf.name_scope("discriminator_loss"):
            if(label_smoothing):
                delta = 0.3
                perturbation = tf.reshape(tf.random.uniform(real_labels.shape, minval=-delta, maxval=delta, dtype=tf.float32, seed=global_seed), [-1, 1])
                added_perturbation = tf.concat([perturbation, -perturbation], axis=1)

                smoothed_fakes = tf.one_hot(fake_labels, depth=2)
                smoothed_reals = tf.clip_by_value(tf.one_hot(real_labels, depth=2) + added_perturbation, 0.0, 1.0)
                
#                with tf.Session() as sess:
#                    print(sess.run(smoothed_reals))
#                    print("\n")
#                    print(sess.run(tf.one_hot(real_labels, depth=2) + added_perturbation))
#                    sys.exit(0)
                    
                fake_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_fakes, logits=fake_out)
                real_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_reals, logits=real_out)
            else:
                fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fake_labels, logits=fake_out)
                real_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_labels, logits=real_out)
            total_loss = tf.reduce_mean(fake_loss + real_loss)
        return total_loss

    def train_op(self, loss, learning_rate, beta1, beta2, var_list, scope):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):  # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss, global_step, var_list)

        return train_op, global_step


