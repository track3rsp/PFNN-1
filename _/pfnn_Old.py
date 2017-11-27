#dognn
import sys
import numpy as np
import random
import tensorflow as tf

tf.set_random_seed(777)  # reproducibility

data = np.float32(np.loadtxt('Data.txt'))
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

num_joint = 21
num_trajectory = 12
num_style = 8
offset = 3
jointNeurons = 6*num_joint  #pos, vel, trans rot vel magnitudes
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


Xmean.tofile('./nn/data/Xmean.bin')
Ymean.tofile('./nn/data/Ymean.bin')
Xstd.tofile('./nn/data/Xstd.bin')
Ystd.tofile('./nn/data/Ystd.bin')


input_x = np.concatenate((X,P),axis = 1) #input of nn, including X and P
input_y = Y


input_size  = input_x.shape[1]
output_size = input_y.shape[1]

number_example =input_x.shape[0]
print("Data is processed")


""" Phase Function Neural Network """

"""input of nn"""
X_nn = tf.placeholder(tf.float32, [None, input_size], name='x-input')
Y_nn = tf.placeholder(tf.float32, [None, output_size], name='y-input')


"""initialize parameters in phase function i.e. alpha and beta 
alpha for calculating weights in nn
beta for calculating bias in nn"""
def initial_alpha(shape, rng=np.random, gamma=0.01):
    alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
    alpha = np.asarray(
        rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
        dtype=np.float32)
    return tf.convert_to_tensor(alpha, dtype = tf.float32)

def initial_beta(shape):
    return tf.zeros(shape,tf.float32)

""" cubic function to calculate the weights and bias in nn"""
def cubic(y0, y1, y2, y3, mu):
    return (
        (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
        (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
        (-0.5*y0+0.5*y2)*mu +
        (y1))

"""initialize alpha and beta in phase function"""
#some fixed parameters 
nslices = 4                             # number of control points in phase function
rng = np.random.RandomState(23456)
keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing


alpha0 = tf.Variable(initial_alpha((nslices, 512, input_size-1), rng=rng, gamma=0.01), name='alpha0') 
beta0 = tf.Variable(initial_beta((nslices, 512)), name='beta0')

alpha1 = tf.Variable(initial_alpha((nslices, 512, 512), rng=rng, gamma=0.01), name='alpha1')
beta1 = tf.Variable(initial_beta((nslices, 512)), name='beta1')

alpha2 = tf.Variable(initial_alpha((nslices, output_size, 512), rng=rng, gamma=0.01), name='alpha2')
beta2 = tf.Variable(initial_beta((nslices, output_size)), name='beta2')



"""calculate the index and weights in phase function """
#index
pscale = nslices * X_nn[:,-1]
pamount = pscale % 1.0
pindex_1 = tf.cast(pscale, 'int32') % nslices
pindex_0 = (pindex_1-1) % nslices
pindex_2 = (pindex_1+1) % nslices
pindex_3 = (pindex_1+2) % nslices

#weight
bamount = tf.expand_dims(pamount, 1) # expand 1 dimension [n*1]
Wamount = tf.expand_dims(bamount, 1) # expand 2 dimension [n*1*1], 


"""initialize weights and bias in nn with CUBIC function tf.nn.embedding_lookup is a function to search table"""
W0 = cubic(tf.nn.embedding_lookup(alpha0,pindex_0), tf.nn.embedding_lookup(alpha0,pindex_1), 
           tf.nn.embedding_lookup(alpha0,pindex_2), tf.nn.embedding_lookup(alpha0,pindex_3), Wamount)
# W0: ?*512*342 

W1 = cubic(tf.nn.embedding_lookup(alpha1,pindex_0), tf.nn.embedding_lookup(alpha1,pindex_1), 
           tf.nn.embedding_lookup(alpha1,pindex_2), tf.nn.embedding_lookup(alpha1,pindex_3), Wamount)
W2 = cubic(tf.nn.embedding_lookup(alpha2,pindex_0), tf.nn.embedding_lookup(alpha2,pindex_1), 
           tf.nn.embedding_lookup(alpha2,pindex_2), tf.nn.embedding_lookup(alpha2,pindex_3), Wamount)

b0 = cubic(tf.nn.embedding_lookup(beta0,pindex_0), tf.nn.embedding_lookup(beta0,pindex_1), 
           tf.nn.embedding_lookup(beta0,pindex_2), tf.nn.embedding_lookup(beta0,pindex_3), bamount)
b1 = cubic(tf.nn.embedding_lookup(beta1,pindex_0), tf.nn.embedding_lookup(beta1,pindex_1), 
           tf.nn.embedding_lookup(beta1,pindex_2), tf.nn.embedding_lookup(beta1,pindex_3), bamount)
b2 = cubic(tf.nn.embedding_lookup(beta2,pindex_0), tf.nn.embedding_lookup(beta2,pindex_1), 
           tf.nn.embedding_lookup(beta2,pindex_2), tf.nn.embedding_lookup(beta2,pindex_3), bamount)


#structure of nn
H0 = X_nn[:,:-1] #input of nn     dims:  ?*342
H0 = tf.expand_dims(H0, -1)       #dims: ?*342*1
H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

b0 = tf.expand_dims(b0, -1)      #dims:  ?*512*1
H1 = tf.matmul(W0, H0) + b0      #dims:  ?*512*342 mul ?*342*1 = ?*512*1
H1 = tf.nn.elu(H1)               #get 1th hidden layer with 'ELU' funciton
H1 = tf.nn.dropout(H1, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

b1 = tf.expand_dims(b1, -1)       #dims: ?*512*1
H2 = tf.matmul(W1, H1) + b1       #dims: ?*512*512 mul ?*512*1 = ?*512*1
H2 = tf.nn.elu(H2)                #get 2th hidden layer with 'ELU' funciton
H2 = tf.nn.dropout(H2, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

b2 = tf.expand_dims(b2, -1)       #dims: ?*311*1
H3 = tf.matmul(W2, H2) + b2       #dims: ?*311*512 mul ?*512*1 =?*311*1
H3 = tf.squeeze(H3, -1)           #dims: ?*311


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
loss = cost + regularization_penalty(alpha0, alpha1, alpha2, 0.01)


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

#saver for saving the variables
saver = tf.train.Saver()



"""save 50 weights and bias for precomputation in real-time """
def save_network(alpha0, alpha1, alpha2, beta0, beta1, beta2, num_points, epoch):
    nslices = 4
    
    for i in range(num_points):
        """calculate the index and weights in phase function """
        pscale = nslices*(float(i)/50)
        #weight
        pamount = pscale % 1.0
        #index
        pindex_1 = int(pscale) % nslices
        pindex_0 = (pindex_1-1) % nslices
        pindex_2 = (pindex_1+1) % nslices
        pindex_3 = (pindex_1+2) % nslices
        
        
        """initialize weights and bias in nn with CUBIC function tf.nn.embedding_lookup is a function to search table"""
        W0 = cubic(alpha0[pindex_0],alpha0[pindex_1],alpha0[pindex_2],alpha0[pindex_3],pamount)
        W1 = cubic(alpha1[pindex_0],alpha1[pindex_1],alpha1[pindex_2],alpha1[pindex_3],pamount)
        W2 = cubic(alpha2[pindex_0],alpha2[pindex_1],alpha2[pindex_2],alpha2[pindex_3],pamount)
        
        b0 = cubic(beta0[pindex_0],beta0[pindex_1],beta0[pindex_2],beta0[pindex_3],pamount)
        b1 = cubic(beta1[pindex_0],beta1[pindex_1],beta1[pindex_2],beta1[pindex_3],pamount)
        b2 = cubic(beta2[pindex_0],beta2[pindex_1],beta2[pindex_2],beta2[pindex_3],pamount)

        
        W0.tofile('./nn/W0_%03i.bin' % i)
        W1.tofile('./nn/W1_%03i.bin' % i)
        W2.tofile('./nn/W2_%03i.bin' % i)
        b0.tofile('./nn/b0_%03i.bin' % i)
        b1.tofile('./nn/b1_%03i.bin' % i)
        b2.tofile('./nn/b2_%03i.bin' % i)
        
        if(epoch % 20 == 0):
            number = str(np.int(epoch / 20))
            W0.tofile('./nn' + '/nn'+ number + '/W0_%03i.bin' % i)
            W1.tofile('./nn' + '/nn'+ number + '/W1_%03i.bin' % i)
            W2.tofile('./nn' + '/nn'+ number + '/W2_%03i.bin' % i)
            b0.tofile('./nn' + '/nn'+ number + '/b0_%03i.bin' % i)
            b1.tofile('./nn' + '/nn'+ number + '/b1_%03i.bin' % i)
            b2.tofile('./nn' + '/nn'+ number + '/b2_%03i.bin' % i)




#start to train
print('Learning started..')
batch_size = 32
training_epochs = 200
I = np.arange(number_example)
error = np.ones(training_epochs)
for epoch in range(training_epochs):
    rng.shuffle(I)
    avg_cost = 0
    total_batch = int(number_example / batch_size)
    print("total_batch:", total_batch)
    for i in range(total_batch):
        index_train = I[i*batch_size:(i+1)*batch_size]
        batch_xs = input_x[index_train]
        batch_ys = input_y[index_train]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7}
        l,c, _, = sess.run([loss,cost, optimizer], feed_dict=feed_dict)
        avg_cost += l / total_batch
        
        if i % 1000 == 0:
            print(i, "loss:", l, "cost:", c)
    
    save_path = saver.save(sess, "./nn/model/model.ckpt")
    save_network(sess.run(alpha0), sess.run(alpha1), sess.run(alpha2), sess.run(beta0), sess.run(beta1), sess.run(beta2), 50, epoch)
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_cost))
    error[epoch] = avg_cost
    error.tofile("./nn/model/error.bin")
print('Learning finished!')
#-----------------------------above is model training----------------------------------


