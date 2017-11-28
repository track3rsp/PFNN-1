#using phase as input
import numpy as np
import tensorflow as tf
import os.path

tf.set_random_seed(23456)  # reproducibility


""" Load Data """

database = np.load('database.npz')
X = database['Xun']
Y = database['Yun']
P = database['Pun']
num_p = 1                             #number of copy of phase
P = P[:,np.newaxis]
P_ex = np.ones((len(P),num_p))*P

X = np.concatenate((X,P_ex),axis = 1) #input of nn, including X and P

print(X.shape, Y.shape)


""" Calculate Mean and Std """

Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

j = 31
w = ((60*2)//10)

Xstd[w*0:w* 1] = Xstd[w*0:w* 1].mean() # Trajectory Past Positions
Xstd[w*1:w* 2] = Xstd[w*1:w* 2].mean() # Trajectory Future Positions
Xstd[w*2:w* 3] = Xstd[w*2:w* 3].mean() # Trajectory Past Directions
Xstd[w*3:w* 4] = Xstd[w*3:w* 4].mean() # Trajectory Future Directions
Xstd[w*4:w*10] = Xstd[w*4:w*10].mean() # Trajectory Gait

""" Mask Out Unused Joints in Input """

joint_weights = np.array([
    1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)

Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w*10+j*3*1].mean() / (joint_weights * 0.1) # Pos
Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w*10+j*3*2].mean() / (joint_weights * 0.1) # Vel
Xstd[w*10+j*3*2:-num_p    ] = Xstd[w*10+j*3*2:-num_p    ].mean() # Terrain
Xstd[-num_p:              ] = Xstd[-num_p:              ].mean() # phase

Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
Ystd[3:4] = Ystd[3:4].mean() # Change in Phase
Ystd[4:8] = Ystd[4:8].mean() # Contacts

Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean() # Trajectory Future Positions
Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean() # Trajectory Future Directions

Ystd[8+w*2+j*3*0:8+w*2+j*3*1] = Ystd[8+w*2+j*3*0:8+w*2+j*3*1].mean() # Pos
Ystd[8+w*2+j*3*1:8+w*2+j*3*2] = Ystd[8+w*2+j*3*1:8+w*2+j*3*2].mean() # Vel
Ystd[8+w*2+j*3*2:8+w*2+j*3*3] = Ystd[8+w*2+j*3*2:8+w*2+j*3*3].mean() # Rot

""" Save Mean / Std / Min / Max """
Xmean = np.float32(Xmean)
Xstd = np.float32(Xstd)


Xmean.tofile('./fnn/data/Xmean.bin')
Ymean.tofile('./fnn/data/Ymean.bin')
Xstd.tofile('./fnn/data/Xstd.bin')
Ystd.tofile('./fnn/data/Ystd.bin')


""" Normalize Data """
for i in range(X.shape[0]):
    X[i,:] = (X[i,:]-Xmean) / Xstd

for i in range(Y.shape[0]):
    Y[i,:] = (Y[i,:]-Ymean) / Ystd



input_size = X.shape[1] #we input both X and P
output_size = Y.shape[1]

"""data for training"""
input_x = X
input_y = Y


number_example =input_x.shape[0]
print("Data is processed")


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
W0 = tf.get_variable("W0", shape=[input_size, 512], initializer=tf.contrib.layers.xavier_initializer())
b0 = tf.Variable(initial_beta((1, 512)), name='b0') #b0: 1*512

# W1: 512*512 
W1 = tf.get_variable("W1", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(initial_beta((1, 512)), name='b1') #b0: 1*512


# W2: 512*311
W2 = tf.get_variable("W2", shape=[512, output_size], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(initial_beta((1, output_size)), name='b2') #b0: 1*512




#structure of nn
H0 = X_nn        #input of nn     dims:  ?*342
H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

H1 = tf.matmul(H0, W0) + b0      #dims:  ?*342 mul 342*512 = ?*512
H1 = tf.nn.elu(H1)               #get 1th hidden layer with 'ELU' funciton
H1 = tf.nn.dropout(H1, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'


H2 = tf.matmul(H1, W1) + b1       #dims: ?*512 mul 512*512 = ?*512
H2 = tf.nn.elu(H2)                #get 2th hidden layer with 'ELU' funciton
H2 = tf.nn.dropout(H2, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

H3 = tf.matmul(H2, W2) + b2       #dims: ?*512 mul 512*311 =?*311


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

loss = tf.reduce_mean(tf.square(Y_nn - H3))
loss_regularization = loss + regularization_penalty(W0, W1, W2, 0.01)


#optimizer, learning rate 0.0001
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


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
num_testBatch  = np.int(total_batch/10)
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
        l,l_r, _, = sess.run([loss,loss_regularization, optimizer], feed_dict=feed_dict)
        avg_cost_train += l / num_trainBatch
        if i % 2500 == 0:
            print(i, "trainingloss:", l, "trainingloss_reg:", l_r)
            
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
        if i % 2500 == 0:
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
    weight0 = sess.run(W0).transpose()
    weight1 = sess.run(W1).transpose()
    weight2 = sess.run(W2).transpose()
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









