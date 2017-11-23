import numpy as np
import tensorflow as tf
import PFNNParameter as PFNN
from PFNNParameter import PFNNParameter
import os.path

tf.set_random_seed(23456)  # reproducibility


""" Load Data """

database = np.load('database.npz')
X = database['Xun']
Y = database['Yun']
P = database['Pun']


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
Xstd[w*10+j*3*2:          ] = Xstd[w*10+j*3*2:          ].mean() # Terrain

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
Xmean.tofile('./human/data/Xmean.bin')
Ymean.tofile('./human/data/Ymean.bin')
Xstd.tofile('./human/data/Xstd.bin')
Ystd.tofile('./human/data/Ystd.bin')


""" Normalize Data """
X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd
P = P[:,np.newaxis]


input_size = X.shape[1]+1 #we input both X and P, hence +1
output_size = Y.shape[1]

"""data for training"""
input_x = np.concatenate((X,P),axis = 1) #input of nn, including X and P
input_y = Y


number_example =input_x.shape[0]
print("Data is processed")

#--------------------------------above is dataprocess-------------------------------------




""" Phase Function Neural Network """


"""input of nn"""
X_nn = tf.placeholder(tf.float32, [None, input_size], name='x-input')
Y_nn = tf.placeholder(tf.float32, [None, output_size], name='y-input')


"""parameter of nn"""
rng = np.random.RandomState(23456)
nslices = 4                             # number of control points in phase function
phase = X_nn[:,-1]                      #phase
P0 = PFNNParameter((nslices, 512, input_size-1), rng, phase, 'wb0')
P1 = PFNNParameter((nslices, 512, 512), rng, phase, 'wb1')
P2 = PFNNParameter((nslices, output_size, 512), rng, phase, 'wb2')



keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
#structure of nn
H0 = X_nn[:,:-1] #input of nn     dims:  ?*342
H0 = tf.expand_dims(H0, -1)       #dims: ?*342*1
H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

b0 = tf.expand_dims(P0.bias, -1)      #dims:  ?*512*1
H1 = tf.matmul(P0.weight, H0) + b0      #dims:  ?*512*342 mul ?*342*1 = ?*512*1
H1 = tf.nn.elu(H1)               #get 1th hidden layer with 'ELU' funciton
H1 = tf.nn.dropout(H1, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

b1 = tf.expand_dims(P1.bias, -1)       #dims: ?*512*1
H2 = tf.matmul(P1.weight, H1) + b1       #dims: ?*512*512 mul ?*512*1 = ?*512*1
H2 = tf.nn.elu(H2)                #get 2th hidden layer with 'ELU' funciton
H2 = tf.nn.dropout(H2, keep_prob=keep_prob) #dropout with parameter of 'keep_prob'

b2 = tf.expand_dims(P2.bias, -1)       #dims: ?*311*1
H3 = tf.matmul(P2.weight, H2) + b2       #dims: ?*311*512 mul ?*512*1 =?*311*1
H3 = tf.squeeze(H3, -1)           #dims: ?*311



#this might be the regularization that Dan use
def regularization_penalty(a0, a1, a2, gamma):
    return gamma * (tf.reduce_mean(tf.abs(a0))+tf.reduce_mean(tf.abs(a1))+tf.reduce_mean(tf.abs(a2)))/3

cost = tf.reduce_mean(tf.square(Y_nn - H3))
loss = cost + regularization_penalty(P0.alpha, P1.alpha, P1.alpha, 0.01)


#optimizer, learning rate 0.0001
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


#session

sess = tf.Session()
sess.run(tf.global_variables_initializer())


#saver for saving the variables
saver = tf.train.Saver()


#start to train
print('Learning start..')
batch_size = 32
training_epochs = 200
total_batch = int(number_example / batch_size)
print("totoal_batch:", total_batch)
I = np.arange(number_example)
rng.shuffle(I)
error = np.ones(training_epochs)
for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        index_train = I[i*batch_size:(i+1)*batch_size]
        batch_xs = input_x[index_train]
        batch_ys = input_y[index_train]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7}
        l,c, _, = sess.run([loss,cost, optimizer], feed_dict=feed_dict)
        avg_cost += l / total_batch
        
        if i % 1000 == 0:
            print(i, "loss:", l, "cost:", c)
    
    save_path = saver.save(sess, "./human/model/model.ckpt")
    PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                      (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                      50, 
                      './human/dog/nn'
                     )
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_cost))
    error[epoch] = avg_cost
    error.tofile("./human/model/error.bin")
    
    
    if epoch>0 and epoch%5==0:
        path_human   = './human/weights/humanNN%03i' % epoch
        if not os.path.exists(path_human):
            os.makedirs(path_human)
        PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                          (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                          50, 
                          path_human)
    
#-----------------------------above is model training----------------------------------
