#using phase as input
import numpy as np
import tensorflow as tf
import os.path

tf.set_random_seed(23456)  # reproducibility


#global parameters for dog and human
num_humanjoint  = 31
num_dogjoint    = 27
num_trajectory  = 12

data = np.float32(np.loadtxt('newData.txt'))
frameCount = data.shape[0]
A = []
B = []
C = []
for i in range(frameCount-2):
    if(data[i,0] == data[i+1,0] and data[i,0] == data[i+2,0]):
        A.append(data[i])
        B.append(data[i+1])
        C.append(data[i+2])
A = np.asarray(A)
B = np.asarray(B)
C = np.asarray(C)

num_joint = num_dogjoint
num_trajectory = 12
num_style = 7
offset = 3
jointNeurons = 12*num_joint  #pos, vel, trans rot vel magnitudes
trajectoryNeurons = (8+num_style)*num_trajectory #pos, dir,hei, style
    

#input 
X = np.concatenate(
        (
                B[:,offset+jointNeurons:offset+jointNeurons+trajectoryNeurons], #trajectory pos, dir, hei, style of B
                A[:,offset:offset+jointNeurons]                                 #joint pos, vel, trans rot vel magnitudes of A
        ),axis = 1) 

#get trajecoty positionX,Z velocityX,Z for future trajectory
Traj_out = np.float32(np.zeros((A.shape[0],np.int(num_trajectory/2*4))))
Traj_out_start = np.int(offset+ jointNeurons+ num_trajectory/2*6)
for i in range(np.int(num_trajectory/2)):
    Traj_out[:,i*4:(i+1)*4] = C[:,[Traj_out_start,Traj_out_start+2,Traj_out_start+3,Traj_out_start+5]]
    Traj_out_start += 6
    
Y = np.concatenate(
        (
                Traj_out, 
                B[:,offset:offset+jointNeurons], 
                B[:,offset+jointNeurons+trajectoryNeurons+1:]
        ),axis = 1)

P = B[:,offset+jointNeurons+trajectoryNeurons]
P = P[:,np.newaxis]

Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

for i in range(Xstd.size):
    if (Xstd[i]==0):
        Xstd[i]=1
for i in range(Ystd.size):
    if (Ystd[i]==0):
        Ystd[i]=1
     
X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd


Xmean.tofile('./fnn/dogdata/Xmean.bin')
Ymean.tofile('./fnn/dogdata/Ymean.bin')
Xstd.tofile('./fnn/dogdata/Xstd.bin')
Ystd.tofile('./fnn/dogdata/Ystd.bin')


input_x = np.concatenate((X,P),axis = 1) #input of nn, including X and P
input_y = Y


input_size  = input_x.shape[1]
output_size = input_y.shape[1]

number_example =input_x.shape[0]

print("DogData is processed")
    
    

#--------------------------------above is dataprocess-------------------------------------

""" Phase Function Neural Network """

"""input of nn"""
X_nn = tf.placeholder(tf.float32, [None, input_size], name='x-input')  #?*342
Y_nn = tf.placeholder(tf.float32, [None, output_size], name='y-input') #?*311

def initial_beta(shape):
    return tf.zeros(shape,tf.float32)

rng = np.random.RandomState(23456)
keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing

# W0: 512*342 
W0 = tf.get_variable("W0", shape=[512,input_size], initializer=tf.contrib.layers.xavier_initializer())     #W0: 512*input_size 
b0 = tf.Variable(initial_beta((512,1)), name='b0')                                                         #b0: 512*1

# W1: 512*512 
W1 = tf.get_variable("W1", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())           #W1: 512*512
b1 = tf.Variable(initial_beta((512,1)), name='b1')                                                        #b1: 512*1


# W2: 512*311
W2 = tf.get_variable("W2", shape=[output_size, 512], initializer=tf.contrib.layers.xavier_initializer())   #W2: out_size*512
b2 = tf.Variable(initial_beta((output_size, 1)), name='b2')                                                #b2: out_size*1




#structure of nn
H0 = tf.transpose(X_nn)           #input of nn     dims:  in*?
H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

H1 = tf.matmul(W0, H0) + b0       #dims: hid*in mul in*? = hid*?
H1 = tf.nn.elu(H1)                #get 1th hidden layer with 'ELU' funciton
H1 = tf.nn.dropout(H1, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'


H2 = tf.matmul(W1, H1) + b1       #dims: hid*hid mul hid*? = hid*?
H2 = tf.nn.elu(H2)                #get 2th hidden layer with 'ELU' funciton
H2 = tf.nn.dropout(H2, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

H3 = tf.matmul(W2, H2) + b2       #dims: out*hid mul hid*? =out*?


#loss function with regularizatoin, and regularization rate=0.01
"""
#this is L1 regularization, but maybe different with Dan, so not use
l1_regularizer = tf.contrib.layers.l1_regularizer(
   scale=0.01, scope=None
)
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, alpha0) + tf.contrib.layers.apply_regularization(l1_regularizer, alpha1)
                         +tf.contrib.layers.apply_regularization(l1_regularizer, alpha2)
"""
#this might be the regularization that Dan use
def regularization_penalty(a0, a1, a2, gamma):
    return gamma * (tf.reduce_mean(tf.abs(a0))+tf.reduce_mean(tf.abs(a1))+tf.reduce_mean(tf.abs(a2)))/3

loss = tf.reduce_mean(tf.square(tf.transpose(Y_nn) - H3))
loss_regularization = loss + regularization_penalty(W0, W1, W2, 0.01)


#optimizer, learning rate 0.0001
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_regularization)


#session

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
sess.run(tf.global_variables_initializer()) 
"""

'''
batch_size = 32
training_epochs = 20
al00 = np.ones((training_epochs,input_size, 512))
al01 = np.ones((training_epochs,512, 512))
al02 = np.ones((training_epochs,512, output_size))
be00 = np.ones((training_epochs,1, 512))
be01 = np.ones((training_epochs,1, 512))
be02 = np.ones((training_epochs,1, output_size))
loss00 = np.ones(training_epochs)
'''

#batch size and epoch
batch_size = 32
training_epochs = 200
total_batch = int(number_example / batch_size)
print("totoal_batch:", total_batch)

#randomly select training set
I = np.arange(number_example)
rng.shuffle(I)


#training set and  test set
count_test  = 0
num_testBatch  = np.int(total_batch*count_test)
num_trainBatch = total_batch - num_testBatch
print("training_batch:", num_trainBatch)
print("test_batch:", num_testBatch)

   
#used for saving errorof each epoch
error_train = np.ones(training_epochs)
error_test  = np.ones(training_epochs)


#save network
"""save 50 weights and bias for precomputation in real-time """
def save_network(W0, W1, W2, b0, b1, b2, num_points, path):
    for i in range(num_points):
        W0.tofile(path+'/W0_%03i.bin' % i)
        W1.tofile(path+'/W1_%03i.bin' % i)
        W2.tofile(path+'/W2_%03i.bin' % i)
        b0.tofile(path+'/b0_%03i.bin' % i)
        b1.tofile(path+'/b1_%03i.bin' % i)
        b2.tofile(path+'/b2_%03i.bin' % i)
        

#start to train
print('Learning start..')
for epoch in range(training_epochs):
    avg_cost_train = 0
    avg_cost_test  = 0
    for i in range(num_trainBatch):
        index_train = I[i*batch_size:(i+1)*batch_size]
        batch_xs = input_x[index_train]
        batch_ys = input_y[index_train]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7}
        l, _, = sess.run([loss_regularization, optimizer], feed_dict=feed_dict)
        avg_cost_train += l / num_trainBatch
        if i % 1000 == 0:
            print(i, "trainingloss:", l)
            
    for i in range(num_testBatch):
        if i==0:
            index_test = I[-(i+1)*batch_size: ]
        else:
            index_test = I[-(i+1)*batch_size: -i*batch_size]
        batch_xs = input_x[index_test]
        batch_ys = input_y[index_test]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 1}
        testError = sess.run(loss, feed_dict=feed_dict)
        avg_cost_test += testError / num_testBatch
        if i % 1000 == 0:
            print(i, "testloss:",testError)
    
    #print and save training test error 
    print('Epoch:', '%04d' % (epoch + 1), 'trainingloss =', '{:.9f}'.format(avg_cost_train))
    print('Epoch:', '%04d' % (epoch + 1), 'testloss =', '{:.9f}'.format(avg_cost_test))
    error_train[epoch] = avg_cost_train
    error_test[epoch]  = avg_cost_test
    error_train.tofile("./fnn/model/error_train.bin")
    error_test.tofile("./fnn/model/error_test.bin")
    
    #save model and weights
    save_path = saver.save(sess, "./fnn/model/model.ckpt")
    
    
    """get np.float32 format"""
    weight0 = sess.run(W0)
    weight1 = sess.run(W1)
    weight2 = sess.run(W2)
    bias0 = sess.run(b0)
    bias1 = sess.run(b1)
    bias2 = sess.run(b2)
    save_network(weight0, weight1, weight2, 
                 bias0,   bias1,   bias2, 
                 50, 
                 './fnn/nn'
                )    
    
    #save weights every 5epoch
    if epoch>0 and epoch%5==0:
        path_NN  = './fnn/weights/NN%03i' % epoch
        if not os.path.exists(path_NN):
            os.makedirs(path_NN)
        save_network(weight0, weight1, weight2, 
                      bias0,   bias1,   bias2, 
                      50, 
                      path_NN
                      )  
      
print('Learning Finished!')









