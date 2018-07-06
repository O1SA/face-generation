import unittest
import os 
from shutil import rmtree
from mnist_gan_with_labels import MnistGAN
import pdb
import tensorflow as tf

class MnistGanWithLabelsTest(unittest.TestCase):

    IN_DIR = 'data/mnist_label'

    def test_can_load_data(self):
        n = MnistGAN()

        # start by deleting folder if exists
        if os.path.exists(self.IN_DIR):
            rmtree(self.IN_DIR)

        self.assertIsNone(n.x)
        self.assertIsNone(n.y)

        # npz will be created 
        n.load_data(self.IN_DIR)

        self.assertIsNotNone(n.x)
        self.assertIsNotNone(n.y)

        # x should be within -1..1
        self.assertEqual( n.x.max(),  1  )
        self.assertEqual( n.x.min(), -1 )
        
        # y should be (-1, 11) with last column all zeros 
        self.assertEqual( n.y.shape[1], 11)

        for i in range(n.y.shape[0]):
            self.assertEqual(n.y[i,-1], 0)

        # if npz exists skip download

        n2 = MnistGAN()
        n2.load_data(self.IN_DIR)

        self.assertIsNotNone(n2.x)

    def test_can_get_batches_from_data(self):
        n = MnistGAN()

        # load data 
        n.load_data(self.IN_DIR)
        
        batch_size = 32
        i = 0
        for y, x in n.get_batches(batch_size) :
            # last batch could be less than 32
            # I will only check the 100 first batches 
            if i < 100:
                # features shape 
                self.assertEqual(x.shape, (batch_size, 28, 28, 1) )

                # labels shape
                self.assertEqual(y.shape, (batch_size, 11))

            # should never return 0 elements
            self.assertGreater(x.shape[0], 0)

            # elements should be scaled adequately
            self.assertEqual(x.max(), 1 )
            self.assertEqual(x.min(), -1)

            # last column of labels should always be zero
            # will check only the first element of the batch
            self.assertEqual(y[0, -1], 0)

            i+=1 

    def test_can_create_input(self):
        n = MnistGAN()

        # create inputs
        input_real, input_z, label_real, label_fake, learning_rate, is_train  = n.create_input(100)

        # input_real (-1, 28, 28, 1)
        self.assertEqual(input_real.shape.as_list(), [None, 28, 28, 1])

        # input_z (-1, 100 )
        self.assertEqual(input_z.shape.as_list(), [None,100])

        # label_real (-1, 11)
        self.assertEqual(label_real.shape.as_list(), [None, 11])

        # label_fake (-1, 11)
        self.assertEqual(label_fake.shape.as_list(), [None,11])

    def test_generator(self):
        
        # create network 
        n = MnistGAN()
        
        # create inputs 
        input_real, input_z, label_real, label_fake, learning_rate, is_train  = n.create_input(100)

        # generator 
        out, logits = n.generator(input_z, label_real, is_train)

        # verify shape of outputs 
        self.assertEqual(out.shape.as_list(), [None, 28, 28, 1])
        self.assertEqual(logits.shape.as_list(), [None, 28, 28, 1])

        out_by_tensor_name = tf.get_default_graph().get_tensor_by_name('generator/out:0')
        self.assertEqual(out_by_tensor_name.shape.as_list(), [None, 28, 28, 1])
        
    def test_discriminator(self):
        # create network 
        n = MnistGAN()

        # create inputs 
        input_real, input_z, label_real, label_fake, learning_rate, is_train  = n.create_input(100)

        # create discriminator with real data
        out, logits = n.discriminator(input_real,is_train)

        # check shape of outputs
        self.assertEqual(out.shape.as_list(), [None, 11])
        self.assertEqual(logits.shape.as_list(), [None,11])

        # verify that we can get tensor by name
        out_by_tensor_name  = tf.get_default_graph().get_tensor_by_name('discriminator/out:0')
        self.assertEqual(out_by_tensor_name.shape.as_list(), [None,11])

    def test_create_loss(self):
        # create network
        n =MnistGAN()
        test_graph = tf.Graph()
        with test_graph.as_default():
            input_real, input_z, label_real, label_fake, learning_rate, is_train  = n.create_input(100)
            d_loss, g_loss = n.create_loss(input_real, input_z, label_real, label_fake, is_train)
            self.assertEqual(d_loss.shape.as_list(), []) # constant 
            self.assertEqual(g_loss.shape.as_list(), []) # constant 