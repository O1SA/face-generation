import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
#from sklearn.model_selection import train_test_split
import os
import numpy as np
import pdb
from shutil import rmtree
import argparse

class MnistGAN():
    """
    MNIST GAN with labels 
    """
    
    IMAGE_MAX_VALUE = 255
    
    def __init__(self):
        self.x = None
        self.y = None 
        self.loaded_file = None 
        
    def load_data(self, in_dir):
        if self.x is not None and self.y is not None:
            # data already loaded
            return 
            
        # check if file exists
        # expect that if one of the files exists both will
        if not os.path.exists(in_dir + '/mnist_label_gan.npz'):
            #pdb.set_trace()
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            
            # combine train & test 
            n = x_train.shape[0] + x_test.shape[0]
            x = np.zeros((n, x_train.shape[1], x_train.shape[2], 1), dtype=np.float32)
            y = np.zeros((n,), dtype=np.uint8)
            
            # scale to -1/1 (output of tanh for generator)
            x[:x_train.shape[0], :, :, :] = (x_train.reshape( x_train.shape + (1,)) / self.IMAGE_MAX_VALUE  - 0.5)  * 2 
            x[x_train.shape[0]:, :, :, :] = (x_test.reshape(  x_test.shape + (1,))  / self.IMAGE_MAX_VALUE  - 0.5)  * 2 
            
            y[:x_train.shape[0],] = y_train
            y[x_train.shape[0]:,] = y_test   
            
            # one hot encode labels
            lb = LabelBinarizer()
            y_lb = lb.fit_transform(y)
            
            # add is fake to labels 
            y = np.zeros((y_lb.shape[0], y_lb.shape[1] + 1))
            y[:,:y_lb.shape[1]] = y_lb
            
            # shuffle data and save to .npz locally
            idx = np.random.permutation(x.shape[0])

            # check if folder exists. creates it if not
            if not os.path.exists(in_dir):
                os.mkdir(in_dir)

            # save to npz
            np.savez(in_dir + '/mnist_label_gan.npz', x = x[idx], y = y[idx])
            
        # data should exists
        f = np.load(in_dir + '/mnist_label_gan.npz')
        self.x = f['x']
        self.y = f['y']
        
    def get_batches(self, batch_size):
        """ generator for getting batches """
        # shuffle data
        idx = np.random.permutation(self.x.shape[0])
        x = self.x[idx]
        y = self.y[idx]

        # will only return full batches
        # only full batches
        for i in range( x.shape[0] // batch_size):
            start = i * batch_size
            end = (i+1) * batch_size

            x_b = x[start:end]
            y_b = y[start:end]

            if x_b.shape[0] > 0:
                yield y_b, x_b

    def generator(self, z, label, is_train, reuse = False):
        """
        input random noise                              (-1, z_dim)
        concat label                                    (-1, z_dim + 10 + 1)
        reshape with fully connected layer              (-1, 7, 7, 256)
        transpose conv. layer #1, strides=2             (-1, 14, 14, 128)
        transpose conv. layer #2, strides=1             (-1, 14, 14, 64)
        transpose conv. layer #3, strides=2             (-1, 28, 28, 1)
        """
        alpha = 0.2
        with tf.variable_scope('generator', reuse = reuse):
            z_concat = tf.concat([z,tf.cast(label, tf.float32)], 1)
            
            x = tf.layers.dense(z_concat, units=7*7*256, activation=None)
            x = tf.reshape(x, shape=[-1,7,7,256])
            x = tf.layers.batch_normalization(x,training = is_train) 
            x = tf.maximum(alpha * x, x)
            
            x1 = tf.layers.conv2d_transpose(x,filters=128, kernel_size=5, strides=2, padding='same')
            x1 = tf.layers.batch_normalization(x1,training = is_train)
            x1 = tf.maximum(alpha * x1, x1)
            
            x2 = tf.layers.conv2d_transpose(x1,filters=64, kernel_size=5, padding='same')
            x2 = tf.layers.batch_normalization(x2,training = is_train)
            x2 = tf.maximum(alpha * x2, x2) 
            
            logits = tf.layers.conv2d_transpose(x2,filters=1,kernel_size=5,strides=2, padding='same')
            out = tf.tanh(logits, name='out')
            
            return out, logits
        
    def discriminator(self, image, is_train, reuse = False):
        """
        image                                           (-1, 28, 28, 1)
        conv. layer #1, strides=2                       (-1, 14, 14, 64)
        conv. layer #2, strides=1                       (-1, 14, 14, 128)
        conv. layer #3, strides=1                       (-1, 7,  7,  256)
        reshape                                         (-1, 7*7*256)
        fully connected layer, logits                   (-1, 10+1)
        sigmoid                                         (-1, 10+1)
        """
        
        alpha = 0.2
        with tf.variable_scope('discriminator', reuse = reuse):
        
            x = tf.layers.conv2d(image, filters=64, kernel_size=5, strides=2, padding='same')
            x = tf.maximum(alpha * x, x) 
            
            x1 = tf.layers.conv2d(x, filters=128, kernel_size=5, padding='same')
            x1 = tf.layers.batch_normalization(x1,training=is_train) 
            x1 = tf.maximum(alpha * x1, x1)
            
            x2 = tf.layers.conv2d(x1, filters=256, kernel_size=5, strides=2, padding='same')
            x2 = tf.layers.batch_normalization(x2,training=is_train) 
            x2 = tf.maximum(alpha * x2, x2) 
            
            x2_reshape = tf.reshape(x2,shape=[-1,7*7*256])
            
            logits = tf.layers.dense(x2_reshape, units=10+1, activation=None)
            
            out = tf.sigmoid(logits, name ='out')
            
            return out, logits
        
    def create_opt(self, d_loss, g_loss, learning_rate, beta1):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
            
        return d_train_opt, g_train_opt
        
    def create_loss(self, input_real, input_z, label_real, label_fake, is_train):
        g_out, g_logits = self.generator(input_z, label_real, is_train)
        d_out_real, d_logits_real = self.discriminator(input_real, is_train)
        d_out_fake, d_logits_fake = self.discriminator(g_out, is_train, reuse = True)
        
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, 
                                                    labels=  tf.cast(label_real,tf.float32)))    
        
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake,
                                                    labels = tf.cast(label_fake,tf.float32)))
                                                    
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, 
                                                    labels = tf.cast(label_real, tf.float32 )))
                                        
        d_loss = d_loss_real + d_loss_fake
                                                    
        return d_loss, g_loss
   
    def create_input(self, z_dim):
        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        
        input_real = tf.placeholder(tf.float32, shape=(None,28,28,1), name='input_real')
        input_z = tf.placeholder(tf.float32, shape=(None,z_dim), name='input_z')
        label_real = tf.placeholder(tf.int8, shape=(None, 11), name='label_real')
        label_fake = tf.placeholder(tf.int8, shape=(None, 11), name='label_fake')
        
        return input_real, input_z, label_real, label_fake, learning_rate, is_train
        
    def train(self, epoch_count, batch_size, z_dim, lrate, beta1, loss_each, image_each, log_dir, save, in_dir, out_dir):
        """ Train the GAN """

        # make sure data is loaded
        self.load_data(in_dir)
        
        # create gan graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            # inputs
            input_real, input_z, label_real, label_fake, learning_rate, is_train = self.create_input(z_dim)
            
            # losses: generator & discriminator
            d_loss, g_loss = self.create_loss(input_real, input_z, label_real, label_fake, is_train)
            
            # optimizers
            d_opt, g_opt = self.create_opt(d_loss, g_loss, learning_rate, beta1)
            
            # placeholder for tensorboard image summary 
            img_sample = tf.placeholder(tf.float32, shape=[None, 28, 28, 1] )
            
            # tensorboard summaries
            disc_summ = tf.summary.scalar('discriminator loss', d_loss)
            gen_summ = tf.summary.scalar('generator loss', g_loss)
            summ_images = tf.summary.image('train images', img_sample, 1)
            summ = tf.summary.merge([disc_summ, gen_summ])
            
            # generator out 
            out = train_graph.get_tensor_by_name('generator/out:0')
        
        it = 0
        with tf.Session(graph=train_graph) as sess:
        
            # initialize variables
            sess.run(tf.global_variables_initializer())
            
            # tensorflow save
            saver = tf.train.Saver() 
            
            # create tensorboard writer
            writer = self._create_writer(log_dir, sess) 
            
            # label fake
            lbl_fake = np.zeros((batch_size, 10 + 1))
            lbl_fake[:,-1] = np.ones((batch_size, 1)).reshape((batch_size,))
            
            for epoch_i in range(epoch_count):
            
                print("Running epoch {}/{}...".format(epoch_i+1, epoch_count) )
    
                for y_real, x_real in self.get_batches(batch_size):

                    batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                    
                    feed = {learning_rate: lrate, 
                            is_train: True, 
                            input_real: x_real, 
                            label_real: y_real,
                            input_z: batch_z,
                            label_fake: lbl_fake}
                    
                    _ = sess.run(d_opt, feed_dict = feed)                               
                    _ = sess.run(g_opt, feed_dict = feed)
                                                    
                    if it % loss_each == 0:
                        # log summaries to tensorboard on current batch
                        s = sess.run(summ, feed_dict={input_real: x_real, label_real: y_real, input_z: batch_z, label_fake: lbl_fake, is_train: False}) 
                        writer.add_summary(s, it)
                        
                    if it % image_each == 0:
                        # log image sample to tensorboard
                        # random number from 0-9
                        id = np.random.random_integers(10) # btwn 0-9
                        sample_lbl = np.zeros((1,11), dtype=np.uint8)
                        sample_lbl[0,id] = 1 
                        sample_z = np.random.uniform(-1, 1, size=[1, z_dim])
                        
                        img_arr = sess.run(out, feed_dict={input_z: sample_z,
                                                           label_real: sample_lbl, 
                                                           is_train: False })
                 
                        # img_arr is between -1 and 1 (output of tanh)
                        img_arr_reshape = np.reshape(img_arr, (1,28,28,1))
                        s = sess.run(summ_images, feed_dict={img_sample: img_arr_reshape})
                        writer.add_summary(s, it) 
                        
                    it+=1
                    
            # save results before closing session 
            if save:
                saver.save(sess, out_dir + '/mnist_gan_label.ckpt')
             
    def load_graph(self, save_file):
        self.loaded_file  = save_file 
        self.loaded_graph = tf.Graph()
        with self.loaded_graph.as_default():
            self.loader = tf.train.import_meta_graph(save_file + '.meta')
            self.input_z = self.loaded_graph.get_tensor_by_name('input_z:0')
            self.label_real = self.loaded_graph.get_tensor_by_name('label_real:0')
            self.is_train = self.loaded_graph.get_tensor_by_name('is_train:0')
            self.gen_out = self.loaded_graph.get_tensor_by_name('generator/out:0')
            self.z_dim = self.input_z.get_shape().as_list()[1]
            
    def inference(self, images, save_file):
        """
        generate images with trained model 
        :param images: List of # to generate (0-9)
        :param save_file: Tensorflow saved model path
        """
        
        if self.loaded_file != save_file:
            self.load_graph(save_file)
        
        # for now, session is closed each time
        with tf.Session(graph=self.loaded_graph) as sess:
            self.loader.restore(sess, save_file)
            samples_z = np.random.uniform(-1, 1, size=[ len(images), self.z_dim])

            # 10 numbers + 1 fake 
            samples_lb = np.zeros((len(images), 11))
            for i, n in enumerate(images):
                samples_lb[i, n] = 1

            feed = {self.input_z: samples_z,
                    self.label_real: samples_lb,
                    self.is_train: False}
            
            # output of tanh btwn -1 and 1
            samples = sess.run(self.gen_out, feed_dict=feed)
            
            # return scaled btwn 0-255
            # to do samples not exactly -1 +1 
            return (samples / 2 + 0.5) * self.IMAGE_MAX_VALUE

    # Helpers

    def _create_writer(self, path, sess):
        """
        - Check if folder exits if not creates it and create summary writer with specified name
        """
        if os.path.exists(path):
            rmtree(path)
        os.makedirs(path)
        writer = tf.summary.FileWriter(path, sess.graph)
        return writer
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'MNIST GAN with labels')  
    parser.add_argument("--train", "-t", help="to train mode", action='store_true')
    parser.add_argument("--save", "-s", help="save model after training", action='store_true' )
    parser.add_argument('--epochs', help='set number of epochs (default=2)', default=2)
    parser.add_argument('--batch_size', help='set batch size (default=32)', default=32)
    parser.add_argument('--z_dim', help='set z dim (default=100)', default=100)
    parser.add_argument('--lrate', help='set learning rate (default=0.0002)', default=0.0004)
    parser.add_argument('--beta1', help='set adam optimizer beta_1 (default=0.7)', default=0.6)
    parser.add_argument('--loss_each', help='log loss to tensorboard each (default=10)', default=10 )
    parser.add_argument('--image_each', help='log image to tensorboard each (default=100)', default=100 )
    parser.add_argument('--log_dir', help='set tensorboard log dir (default=log/mnist_label)', default='log/mnist_label')
    parser.add_argument('--in_dir',  help='set data dir (default=data/mnist_label)', default='data/mnist_label')
    parser.add_argument('--out_dir', help='set output dir (default=output)', default='output')

    args = parser.parse_args()
    
    if args.train:
        # run train 

        # create GAN 
        gan = MnistGAN()

        gan.train(int(args.epochs),
                  args.batch_size,
                  args.z_dim,
                  args.lrate, 
                  args.beta1,
                  args.loss_each, 
                  args.image_each,
                  args.log_dir, 
                  args.save,
                  args.in_dir,
                  args.out_dir)

# tensorboard --logdir=log/mnist_label 

# GPU
# floyd login
# floyd run --gpu --tensorboard --data ostamand/datasets/mnist-gan-label/1:mnist_label --env tensorflow-1.8 "python mnist_gan_with_labels.py --train --save --epochs 5 --log_dir /output/log --out_dir /output --in_dir /mnist_label"


