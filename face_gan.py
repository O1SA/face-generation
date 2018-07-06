import tensorflow as tf 
import numpy as np
import argparse
import helper
import os
from glob import glob
from shutil import rmtree
import pdb

class FaceGAN():
    
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.WARN) 
        
    def create_inputs(self, z_dim):
        is_train = tf.placeholder(tf.bool, name = 'is_train')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        
        input_real = tf.placeholder(tf.float32, 
                                    shape=(None, 28, 28, 3), 
                                    name='input_real')

        input_z = tf.placeholder(tf.float32, 
                                 shape=(None, z_dim), 
                                 name='input_z')
                                  
        return input_real, input_z, is_train, learning_rate
        
    def create_loss(self, input_real, input_z, is_train):
        g_model, g_logits = self.generator(input_z, is_train)
        
        d_model_real, d_logits_real = self.discriminator(input_real)
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse = True)
        
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, 
                                                    labels=0.9*tf.ones_like(d_model_real)))
        
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, 
                                                    labels=tf.zeros_like(d_model_fake)))
        
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, 
                                                    labels=tf.ones_like(d_model_fake)))
        
        d_loss = d_loss_real + d_loss_fake
        
        return d_loss, g_loss
     
    def create_opts(self, d_loss, g_loss, learning_rate, beta1):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

    def generator(self, z, is_train, reuse = False):
        """ Create generator layers
        
        input random noise                          (-1, z_dim)
        reshape with fully connected layer          (-1, 7, 7, 256)
        transpose conv. layer #1, strides=2         (-1, 14, 14, 128)
        transpose conv. layer #2, strides=1         (-1, 14, 14, 64)
        transpose conv. layer #3, strides=2         (-1, 28, 28, 3)
        
        Args:
          z (tf.placeholder): Input noise to the generator
          reuse (bool): Reuse variables of the layer 
          
        Returns: 
          (out, logits): Tuple
        
        """

        alpha = 0.2
        with tf.variable_scope('generator', reuse = reuse):
            x = tf.layers.dense(z, units=7*7*256, activation=None)
            x = tf.reshape(x, shape=[-1,7,7,256])
            x = tf.layers.batch_normalization(x,training = is_train) 
            x = tf.maximum(alpha * x, x) 
            
            x1 = tf.layers.conv2d_transpose(x,filters=128, kernel_size=5, strides=2, padding='same')
            x1 = tf.layers.batch_normalization(x1,training = is_train)
            x1 = tf.maximum(alpha * x1, x1) 
            
            x2 = tf.layers.conv2d_transpose(x1,filters=64, kernel_size=5, padding='same')
            x2 = tf.layers.batch_normalization(x2,training = is_train)
            x2 = tf.maximum(alpha * x2, x2) 
            
            logits = tf.layers.conv2d_transpose(x2,filters=3,kernel_size=5,strides=2, padding='same')
            out = tf.tanh(logits, name='out')
            
            return out, logits
            
    def discriminator(self, images, reuse = False):
        """ Create discriminator layers
        
        image                               (-1, 28, 28, 3)
        conv. layer #1, strides=2           (-1, 14, 14, 64)
        conv. layer #2, strides=1           (-1, 14, 14, 128)
        conv. layer #3, strides=2           (-1, 7, 7, 256)
        reshape                             (-1, 7*7*256)
        fully connected layer, logits       (-1, 1)
        sigmoid                             (-1, 1)
        
        Args:
          images (tf.placeholder): Input images to the discriminator
          reuse (bool): Reuse variables of the layer
          
        Returns: 
          (out, logits): Tuple
        
        """
        
        alpha = 0.2
        with tf.variable_scope('discriminator', reuse = reuse):
            x = tf.layers.conv2d(images, filters=64, kernel_size=5, strides=2, padding='same')
            x = tf.maximum(alpha * x, x) 
            
            x1 = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same')
            x1 = tf.layers.batch_normalization(x1,training=True) 
            x1 = tf.maximum(alpha * x1, x1) 
            
            x2 = tf.layers.conv2d(x1, filters=256, kernel_size=5, padding='same')
            x2 = tf.layers.batch_normalization(x2,training=True) 
            x2 = tf.maximum(alpha * x2, x2) 
            
            x2_reshape = tf.reshape(x2,shape=[-1,7*7*256])
            
            logits = tf.layers.dense(x2_reshape, units=1, activation=None)
            out = tf.sigmoid(logits, name = 'out')
            
            return out, logits
        
    def train(self, epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches,
              loss_each, 
              image_each, 
              log_dir,
              save, 
              out_dir, 
              restore = None):
        """Train GAN
        
        Args:
          epoch_count (int): Number of epochs to run
          batch_size (int): Size of train batches
          z_dim (int): Random noise input dimension 
          learning_rate (float): Learning rate
          beta1 (float): Adam optimizer beta1
          get_batches (generator function): Generator function to get batches of sise batch_size
          loss_each (int): Log loss to TensorBoard each number of iterations
          image_each (int): Log sample image to TensorBoard each number of iterations
          log_dir (string): Path to directory where to save TensorBoard files
          save (bool): Save checkpoint at the end of training 
          out_dir (string): Path to directory where to save outputs
          restore (string or None): Path to restore checkpoint or None if not used
          
        Returns:
          None
        """
        train_graph = tf.Graph()
        with train_graph.as_default():            
            # create placeholders
            input_real, input_z, is_train, lr  = self.create_inputs(z_dim)
            
            # create loss
            d_loss, g_loss = self.create_loss(input_real, input_z, is_train)
            
            # create optimizers
            d_opt, g_opt = self.create_opts(d_loss, g_loss, lr, beta1)
            
            # tensorboard logging
            sample_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
            disc_summ = tf.summary.scalar('discriminator loss', d_loss)
            gen_summ = tf.summary.scalar('generator loss', g_loss)
            summ_images = tf.summary.image('train images', sample_image, 1)
            summ = tf.summary.merge([disc_summ, gen_summ])
            
            out = train_graph.get_tensor_by_name('generator/out:0')
        
        it = 0
        with tf.Session(graph=train_graph) as sess: 
            pdb.set_trace()

            # tensorflow save
            saver = tf.train.Saver() 
            
            # initialize variables
            # start from scratch or restore existing 
            if restore is None:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, restore)
                
            # create tensorboard writer
            writer = self._create_writer(log_dir, sess) 
            
            for epoch_i in range(epoch_count):
            
                print("Running epoch {}/{}...".format(epoch_i+1, epoch_count) )
                
                for batch_images in get_batches(batch_size):
                
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                    
                    feed = {input_real: batch_images*2,
                            input_z: batch_z, 
                            lr: learning_rate,
                            is_train: True}
                    
                    _ = sess.run(d_opt, feed_dict=feed)
                    _ = sess.run(g_opt, feed_dict=feed)
                                                   
                    if it % loss_each == 0:
                        # log loss to tensorboard
                        s = sess.run(summ, feed_dict={input_real: batch_images*2, 
                                                      input_z: batch_z, 
                                                      is_train: False}) 
                        writer.add_summary(s, it)
                        
                    if it % image_each == 0:
                        # log sample image to tensorboard
                        sample_z = np.random.uniform(-1, 1, size=[1, z_dim])
                        img_arr = sess.run(out,feed_dict={input_z: sample_z, is_train: False})                     
                        img_arr_reshape = np.reshape(img_arr, (1,28,28,3))
                        s = sess.run(summ_images, feed_dict={sample_image: img_arr_reshape, is_train: False})
                        writer.add_summary(s, it)   
                        
                    it+=1
                    
            # save results before closing session 
            if save:
                saver.save(sess, out_dir + '/face_gan.ckpt')

            print('Done.')
                    
    def inference(self, n_images, save_file ):
        """ Infer n images from the saved graph
        
        Args:
          n_images (int): Number of images to inference
          save_file (string): Path to TensorFlow saver
          
        Returns:
          A numpy array with images scaled from -1 to 1.
          Shape is (n_images, 28, 28, 3)
        
        """
        
        # for now create graph & restore each time 
        loaded_graph = tf.Graph()
        with loaded_graph.as_default() as g:
            loader = tf.train.import_meta_graph(save_file + '.meta')
            
            # get tensor from loaded graph
            input_z = g.get_tensor_by_name('input_z')
            is_train = g.get_tensor_by_name('is_train')
            gen_out = g.get_tensor_by_name('generator/out:0')
            z_dim = input_z.get_shape().as_list()[-1]
            
        with tf.Session(graph=loaded_graph) as sess:
            loader.restore(sess, save_file)
            samples_z = np.random.uniform(-1, 1, size=[n_images, z_dim])
            samples = sess.run(gen_out, feed_dict = {input_z: samples_z, is_train: False})
            return samples
                         
    # Helpers
    
    def _create_writer(self, path, sess):
        """ Return tf.summary.FileWriter with specified path & session
        
        Args:
          path (string): Path where tf.summary.FileWriter should save
          sess (tf.Session): Session to use 
        
        Returns:
          tf.summary.FileWriter
        """
        if os.path.exists(path):
            rmtree(path)
        os.makedirs(path)
        writer = tf.summary.FileWriter(path, sess.graph)
        return writer
              
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'FACE GAN')  
    parser.add_argument("--train", "-t", help="to train mode", action='store_true')
    parser.add_argument("--save", "-s", help="save model after training", action='store_true' )
    parser.add_argument('--epochs', help='set number of epochs (default=2)', default=2)
    parser.add_argument('--batch_size', help='set batch size (default=32)', default=32)
    parser.add_argument('--z_dim', help='set z dim (default=100)ju', default=100)
    parser.add_argument('--lrate', help='set learning rate (default=0.0004)', default=0.0004)
    parser.add_argument('--beta1', help='set adam optimizer beta_1 (default=0.7)', default=0.6)
    parser.add_argument('--alpha', help='set leaky relu alpha (default=0.2)', default=0.2)
    parser.add_argument('--loss_each', help='log loss to tensorboard each (default=10)', default=10 )
    parser.add_argument('--image_each', help='log image to tensorboard each (default=100)', default=100 )
    parser.add_argument('--log_dir', help='set tensorboard log dir (default=log/face)', default='log/face')
    parser.add_argument('--in_dir', help='set data dir (default=data/face)', default='data/face')
    parser.add_argument('--out_dir', help='set output dir (default=output)', default='output')
    parser.add_argument('--restore', help='restore file before training', default=None)
    
    args = parser.parse_args()
    
    if args.train:
        # run train 
        
        # create GAN 
        gan = FaceGAN()
        
        # get data
        pdb.set_trace()
        celeba_dataset = helper.Dataset('celeba', glob( args.in_dir + '/*.jpg'))
        print("number of files: {}".format( len(glob(  args.in_dir + '/*.jpg')) ))
        
        gan.train(int(args.epochs),
                  args.batch_size,
                  args.z_dim,
                  args.lrate, 
                  args.beta1,
                  celeba_dataset.get_batches,
                  args.loss_each, 
                  args.image_each,
                  args.log_dir, 
                  args.save,
                  args.out_dir, 
                  args.restore)

# run local with default values
# python face_gan.py --train --save 

# GPU
# floyd login

# with restore ckpt 
# floyd run --gpu --tensorboard --data udacity/datasets/celeba/1:face --data ostamand/datasets/restore-face-gan/1:restore --env tensorflow-1.8 "python face_gan.py --train --save --epochs 5 --log_dir /output/log --out_dir /output --data_dir /face --restore /restore/face_gan.ckpt"

# no restore ckpt
# floyd run --gpu --tensorboard --data udacity/datasets/celeba/1:face --env tensorflow-1.8 "python face_gan.py --train --save --epochs 5 --log_dir /output/log --out_dir /output --data_dir /face"