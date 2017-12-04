#pfnn tensorflow-cpu
import sys
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


tf.set_random_seed(777)  # reproducibility

rng = np.random.RandomState(23456)


'''
t1 = np.array([[1,2],[2,3],[4,5]])

t2 = np.array([[-1],[-2],[-3]])

t3 = t2.transpose()

t4 = t1+t2

'''

'''
nslices = 4
for i in range(50):
    """calculate the index and weights in phase function """
    #index
    pscale = nslices*(float(i)/50)
    pamount = pscale % 1.0
    pindex_1 = int(pscale) % nslices
    pindex_0 = (pindex_1-1) % nslices
    pindex_2 = (pindex_1+1) % nslices
    pindex_3 = (pindex_1+2) % nslices
    
    print(pscale,pamount,pindex_0,pindex_1,pindex_2,pindex_3)
'''


'''
# expand dimension
sess = tf.InteractiveSession()
labels =  [[1,2,3],[2,3,4],[4,3,2]]

x = np.array(labels)

x2 = tf.expand_dims(x, 1)

x3 = tf.expand_dims(x, -1)

x4 = tf.squeeze(x3,-1)
'''

'''
#read->reshape->transpose
for i in range(50):
    W0 = np.fromfile('./pfnn/pfnn11/W0_%03i.bin' % i , dtype=np.float32)
    W0 = np.reshape(W0, (342, 512))
    W0 = np.transpose(W0)
    
    W1 = np.fromfile('./pfnn/pfnn11/W1_%03i.bin' % i , dtype=np.float32)
    W1 = np.reshape(W1, (512, 512))
    W1 = np.transpose(W1)
    
    W2 = np.fromfile('./pfnn/pfnn11/W2_%03i.bin' % i , dtype=np.float32)
    W2 = np.reshape(W2, (512, 311))
    W2 = np.transpose(W2)
    
    
    
    W0.tofile('./pfnn/pfnn11/W0_%03i.bin' % i)
    W1.tofile('./pfnn/pfnn11/W1_%03i.bin' % i)
    W2.tofile('./pfnn/pfnn11/W2_%03i.bin' % i)
    
    print(i);
'''


'''
labels1 = np.array([[1.0,1.0],[2.0,2.0],[3.0,3.0]])
labels1_m = labels1.mean(axis = 0)
labels1_s = labels1.std(axis = 0)

labels2 = labels1.copy()
labels3 = (labels1-labels1_m)/labels1_s

for i in range(labels2.shape[0]):
    labels2[i,:] = (labels2[i,:]-labels1_m) / labels1_s
    
tt = labels2-labels3
'''

'''
W0 = np.fromfile('./pfnn/pfnn_taku/W0_001.bin' , dtype=np.float32)

W00 = np.fromfile('./pfnn/pfnn/W0_001.bin' , dtype=np.float32)
'''


'''
II = np.arange(895130)
I = np.arange(895130)
rng.shuffle(I)
index = I[100:200]
III = II[index]
'''






'''
#test tf model

""" Load Data """

database = np.load('database.npz')
X = database['Xun']
Y = database['Yun']
P = database['Pun']
num_p = 50     #number of copy of phase
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
Xmean.tofile('./fnn1/data/Xmean.bin')
Ymean.tofile('./fnn1/data/Ymean.bin')
Xstd.tofile('./fnn1/data/Xstd.bin')
Ystd.tofile('./fnn1/data/Ystd.bin')


""" Normalize Data """
X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd


input_size = X.shape[1] #we input both X and P
output_size = Y.shape[1]

"""data for training"""
input_x = X
input_y = Y


number_example =input_x.shape[0]
print("Data is processed")


#-----------------------------------------------------------------------------------------------


tf.reset_default_graph()




""" Phase Function Neural Network """

"""input of nn"""
X_nn = tf.placeholder(tf.float32, [None, input_size], name='x-input')  #?*342
Y_nn = tf.placeholder(tf.float32, [None, output_size], name='y-input') #?*311


"""initialize the weight and bias of nn"""
def initial_weight(shape, rng=np.random, gamma=0.01):
    alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
    alpha = np.asarray(
        rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
        dtype=np.float32)
    return tf.convert_to_tensor(alpha, dtype = tf.float32)

def initial_beta(shape):
    return tf.zeros(shape,tf.float32)


rng = np.random.RandomState(23456)
keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing

W0 = tf.Variable(tf.zeros([input_size, 512]), name='W0') # W0: 512*342 
b0 = tf.Variable(initial_beta((1, 512)), name='b0') #b0: 1*512

W1 = tf.Variable(tf.zeros([512, 512]), name='W1') # W0: 512*342 
b1 = tf.Variable(initial_beta((1, 512)), name='b1') #b1: 1*512

W2 = tf.Variable(tf.zeros([512, output_size]), name='W2') # W0: 512*342 
b2 = tf.Variable(initial_beta((1, output_size)), name='b2') #b2: 1*311




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

cost = tf.reduce_mean(tf.square(Y_nn - H3))
loss = cost + regularization_penalty(W0, W1, W2, 0.01)


#optimizer, learning rate 0.0001
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


#session

sess = tf.Session()
sess.run(tf.global_variables_initializer())


"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
sess.run(tf.global_variables_initializer()) 
"""
saver = tf.train.Saver()


checkpoint_dir ="./fnn1/model/model.ckpt"
saver.restore(sess, checkpoint_dir)  

batch_xs = input_x[200:300]
batch_ys = input_y[200:300]
feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 1}

test_W0,test_W1,test_W2,test_H0, test_H1, test_H2, test_H3= sess.run([W0,W1,W2,H0, H1,H2,H3], feed_dict = feed_dict)

'''

''' 
#test initialize
def initial_weight1(shape, rng=np.random):
    alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
    alpha = np.asarray(
        rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
        dtype=np.float32)
    return alpha


rng = np.random.RandomState(23456)
shape = (392,512)
test_i_1 = initial_weight1(shape, rng=rng)
alpha_bound = np.sqrt(600. / np.prod(shape[-2:]))
ttttt = np.prod(shape[-2:])
'''


'''
error_train = np.fromfile('error_train.bin', dtype=np.float64)[:200]


error_test = np.fromfile('error_test.bin', dtype=np.float64)[:200]

x = range(200)


plt.plot(x, error_train, '-', color='r', label="train")
plt.plot(x, error_test,  '-', color='g', label="test")
plt.legend(loc='upper right')

'''
'''
test of fromfile and reshape
sharedAlpha = np.fromfile('./control/alpha1.bin', dtype=np.float32).reshape((4,512,512))
sharedBeta  = np.fromfile('./control/beta1.bin', dtype=np.float32).reshape((4,512))


sharedAlpha.tofile('test.bin')

sharedAlpha_t1 = np.fromfile('test.bin', dtype=np.float32)

sharedAlpha_t2 = sharedAlpha_t1.reshape((4,512,512))

tt             = sharedAlpha - sharedAlpha_t2

'''






