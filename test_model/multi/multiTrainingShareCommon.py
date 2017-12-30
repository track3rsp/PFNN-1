"""
this is used for multi-task training, where we share terrain, 
share trajectory in input/output
common things:
    input:
        trajectory position, direction, terrain
    output:
        trajectory future position and direction
"""
import numpy as np
import tensorflow as tf
import PFNNParameter as PFNN
from PFNNParameter import PFNNParameter
import os.path

tf.set_random_seed(23456)  # reproducibility

#global parameters for dog and human
num_humanjoint  = 31
num_dogjoint    = 21
num_trajectory  = 12

""" Load human data"""
def LoadHuman(filename):
    database = np.load(filename)
    X = database['Xun']
    Y = database['Yun']
    P = database['Pun']
    
    """ Calculate Mean and Std """
    Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
    Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)
    
    j = num_humanjoint
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
    Xmean.tofile('./result/humandata/Xmean.bin')
    Ymean.tofile('./result/humandata/Ymean.bin')
    Xstd.tofile('./result/humandata/Xstd.bin')
    Ystd.tofile('./result/humandata/Ystd.bin')
    
    
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
    print("HumanData is processed")
    return input_x, input_y, input_size, output_size, number_example

"""Load dog data"""
def LoadDog(filename):
    data = np.float32(np.loadtxt(filename))
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
    
    
    Xmean.tofile('./result/dogdata/Xmean.bin')
    Ymean.tofile('./result/dogdata/Ymean.bin')
    Xstd.tofile('./result/dogdata/Xstd.bin')
    Ystd.tofile('./result/dogdata/Ystd.bin')
    
    
    input_x = np.concatenate((X,P),axis = 1) #input of nn, including X and P
    input_y = Y
    
    
    input_size  = input_x.shape[1]
    output_size = input_y.shape[1]
    
    number_example =input_x.shape[0]
    
    
    #get the terrain index for multitask learning
    index_terrain = list(range(3*num_trajectory))
    terrainM_start = 1
    terrainL_start = 6*num_trajectory
    for i in range(num_trajectory):
        index_terrain[i]                  = terrainL_start+1
        index_terrain[i+num_trajectory]   = terrainM_start 
        index_terrain[i+num_trajectory*2] = terrainL_start
        terrainM_start += 6 
        terrainL_start += 2


    #get the trajectory index for multitask learning in the input
    index_inputTrajectory  = list(range(4*num_trajectory))
    for i in range(num_trajectory):
        index_inputTrajectory[i]                  = i*6 
        index_inputTrajectory[i+num_trajectory]   = i*6+2
        index_inputTrajectory[i+2*num_trajectory] = i*6+3
        index_inputTrajectory[i+3*num_trajectory] = i*6+5
    
    #get the future-trajectory index for nultitask learning in the output
    index_outputTrajectory = list(range(2*num_trajectory))
    num_halftrajectory     = np.int(num_trajectory/2)
    for i in range(num_halftrajectory):
        index_outputTrajectory[i]                        = i*4
        index_outputTrajectory[i+ num_halftrajectory]    = i*4+1
        index_outputTrajectory[i+ num_halftrajectory*2]  = i*4+2
        index_outputTrajectory[i+ num_halftrajectory*3]  = i*4+3
    
    print("DogData is processed")
    return input_x, input_y, input_size, output_size, number_example, index_terrain, index_inputTrajectory, index_outputTrajectory
    

input_human, output_human, inputSize_human, outputSize_human, size_human               = LoadHuman('database.npz')
input_dog,   output_dog,   inputSize_dog,   outputSize_dog,   size_dog,  index_terrain, index_inputTrajectory,index_outputTrajectory = LoadDog('Data.txt')
#--------------------------------above is dataprocess-------------------------------------




""" Phase Function Neural Network """

"""input of nn"""
X_human = tf.placeholder(tf.float32, [None, inputSize_human], name='human-input')
Y_human = tf.placeholder(tf.float32, [None, outputSize_human], name='human-output')

X_dog = tf.placeholder(tf.float32, [None, inputSize_dog], name='dog-input')
Y_dog = tf.placeholder(tf.float32, [None, outputSize_dog], name='dog-output')


"""flag for human or dog"""
flag = tf.placeholder(tf.int32)
tensor_1 = tf.constant(1)


"""parameter of nn"""
nslices = 4                             # number of control points in phase function
rng = np.random.RandomState(23456)
phase_human = X_human[:,-1] 
phase_dog   =X_dog[:,-1] 

phase = tf.cond(tf.less(tensor_1, flag), 
                 lambda: phase_human,
                 lambda: phase_dog
                 )
terrainNuerons    = 3 * num_trajectory
trajectoryNuerons = 4 * num_trajectory
half_trajectoryNuerons = np.int(trajectoryNuerons/2)
P0_human  = PFNNParameter((nslices, 512, inputSize_human-1 - terrainNuerons - trajectoryNuerons), rng, phase, 'wb0_human')
P0_dog    = PFNNParameter((nslices, 512, inputSize_dog-1), rng, phase, 'wb0_dog')
P1        = PFNNParameter((nslices, 512, 512), rng, phase, 'wb1')
P2_human  = PFNNParameter((nslices, outputSize_human - half_trajectoryNuerons, 512), rng, phase, 'wb2_human')
P2_dog    = PFNNParameter((nslices, outputSize_dog, 512), rng, phase, 'wb2_dog')

#get terrainweights from P0_dog    
W0_terrain    =  P0_dog.getWeights(index_terrain,-1)
W0_trajectory =  P0_dog.getWeights(index_inputTrajectory,-1)
W2_trajectory =  P2_dog.getWeights(index_outputTrajectory,1)
b2_trajectory =  P2_dog.getBias(index_outputTrajectory,-1)

"""weights and parameter of nn"""
W0 = tf.cond(tf.less(tensor_1, flag), 
             lambda: tf.concat([W0_trajectory,P0_human.weight, W0_terrain], axis = -1),
             lambda: P0_dog.weight
             )

W1 = P1.weight

W2 = tf.cond(tf.less(tensor_1, flag), 
             lambda: tf.concat([P2_human.weight[:,:8,:], W2_trajectory, P2_human.weight[:,8:,:]], axis = 1),
             lambda: P2_dog.weight
             )

b0 = tf.cond(tf.less(tensor_1, flag), 
             lambda: P0_human.bias,
             lambda: P0_dog.bias
             )

b1 = P1.bias

b2 = tf.cond(tf.less(tensor_1, flag), 
             lambda: tf.concat([P2_human.bias[...,:8], b2_trajectory, P2_human.bias[...,8:]], axis = -1),
             lambda: P2_dog.bias
             )



"""structure of nn"""
keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
H0 = tf.cond(tf.less(tensor_1, flag),           #?*inputsize
             lambda: X_human[:,:-1],
             lambda: X_dog[:,:-1]
             )

H0 = tf.expand_dims(H0, -1)                     #?*InSize*1
H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

b0 = tf.expand_dims(b0, -1)                     #?*HidSize*1
H1 = tf.matmul(W0, H0) + b0                     #?*HidSize*InSize mul ?*InSize*1 = ?*HidSize*1
H1 = tf.nn.elu(H1)               
H1 = tf.nn.dropout(H1, keep_prob=keep_prob)

b1 = tf.expand_dims(b1, -1)                     #?*HidSize*1
H2 = tf.matmul(W1, H1) + b1                     #?*521*HidSize mul ?*HidSize*1 = ?*HidSize*1
H2 = tf.nn.elu(H2)                              
H2 = tf.nn.dropout(H2, keep_prob=keep_prob) 


b2 = tf.expand_dims(b2, -1)                     #?*OutSize*1
H3 = tf.matmul(W2, H2) + b2                     #?*OutSize*HidSize mul ?*HidSize*1 = ?*OutSize*1
H3 = tf.squeeze(H3, -1)                         #?*OutSize


"""get the loss function"""

Y = tf.cond(tf.less(tensor_1, flag),  
            lambda:Y_human,
            lambda:Y_dog
            )
regularization = tf.cond(tf.less(tensor_1, flag),  
                         lambda: PFNN.regularization_penalty((P0_human.alpha, P1.alpha, P2_human.alpha),0.01),
                         lambda: PFNN.regularization_penalty((P0_dog.alpha,   P1.alpha, P2_dog.alpha),  0.01)
                         )
loss = tf.reduce_mean(tf.square(Y - H3))
loss_regularization = loss + regularization


"""optimizer, learning rate 0.0001"""
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_regularization)


"""session"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())
"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
sess.run(tf.global_variables_initializer()) 
"""

#saver for saving the variables
saver = tf.train.Saver()


#batch size and epoch
batch_size        = 32
training_epochs   = 200
total_batch_dog   = int(size_dog / batch_size)
total_batch_human = int(size_human/batch_size)
HumanMDog         = np.int(total_batch_human/total_batch_dog) #how many human epoches when one dog epoch

print("total_batch_human:", total_batch_human)
print("total_batch_dog", total_batch_dog)


#randomly select training set
I_dog   =  np.arange(size_dog)
rng.shuffle(I_dog)

I_human =  np.arange(size_human)
rng.shuffle(I_human)

#training set and  test set
count_test  = 0
num_testBatch_human  = np.int(total_batch_human * count_test)
num_trainBatch_human = total_batch_human - num_testBatch_human

num_testBatch_dog  = np.int(total_batch_dog * count_test)
num_trainBatch_dog = total_batch_dog - num_testBatch_dog
print("training_batch_human:", num_trainBatch_human)
print("test_batch_human:", num_testBatch_human)
print("training_batch_dog:", num_trainBatch_dog)
print("test_batch_dog:", num_testBatch_dog)

#used for saving error of each epoch
error_human_train = np.ones(training_epochs)
error_human_test  = np.ones(training_epochs)
error_dog_train   = np.ones(training_epochs)
error_dog_test    = np.ones(training_epochs)

#start training 
print('Learning start..')
for epoch in range(training_epochs):
    human_loss_train = 0
    human_loss_test  = 0
    dog_loss_train   = 0
    dog_loss_test    = 0
    
    #training 
    for i in range(num_trainBatch_dog):
        index_train_dog = I_dog[i*batch_size:(i+1)*batch_size]
        dog_xs   = input_dog[index_train_dog]
        dog_ys   = output_dog[index_train_dog]
        
        for j in range(HumanMDog):
            index_train_human = I_human[(i*HumanMDog+j)*batch_size:(i*HumanMDog+j+1)*batch_size]
            human_xs   =  input_human[index_train_human]
            human_ys   =  output_human[index_train_human]
            #train human
            feed_dict = {X_human: human_xs, Y_human: human_ys, 
                         X_dog: dog_xs, Y_dog: dog_ys, 
                         flag: 2, 
                         keep_prob: 0.7
                         }
            l_human, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
            human_loss_train += l_human / num_trainBatch_human
        #train dog
        feed_dict = {X_human: human_xs, Y_human: human_ys, 
                     X_dog: dog_xs, Y_dog: dog_ys, 
                     flag: 1, 
                     keep_prob: 0.7
                     }
        l_dog, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
        dog_loss_train += l_dog / num_trainBatch_dog
        if i % 1000 == 0:
            print(i, "human_train_loss:",l_human, "dog_train_loss:", l_dog)
                         
    #testing   
    for i in range(num_testBatch_dog):
        if i==0:
            index_test_dog = I_dog[-(i+1)*batch_size:]
        else:
            index_test_dog = I_dog[-(i+1)*batch_size: -i*batch_size]
 
        dog_xs   = input_dog[index_test_dog]
        dog_ys   = output_dog[index_test_dog]
        
        for j in range(HumanMDog):
            if i==0 and j==0:
                index_test_human = I_human[-(i+1)*batch_size:]
            else:
                index_test_human = I_human[-(i*HumanMDog+j+1)*batch_size: -(i*HumanMDog+j)*batch_size]
            
            
            human_xs   =  input_human[index_test_human]
            human_ys   =  output_human[index_test_human]
            #test human
            feed_dict = {X_human: human_xs, Y_human: human_ys, 
                         X_dog: dog_xs, Y_dog: dog_ys, 
                         flag: 2, 
                         keep_prob: 1
                         }
            l_human, _, = sess.run([loss, optimizer], feed_dict=feed_dict)
            human_loss_test += l_human / num_testBatch_human        


        #test dog
        feed_dict = {X_human: human_xs, Y_human: human_ys, 
                     X_dog: dog_xs, Y_dog: dog_ys, 
                     flag: 1, 
                     keep_prob: 1
                     }
        l_dog  = sess.run(loss, feed_dict=feed_dict)
        dog_loss_test += l_dog / num_testBatch_dog
        if i % 1000 == 0:
            print(i, "human_test_loss:",l_human, "dog_test_loss:", l_dog)            
    

    print('Epoch:', '%04d' % (epoch + 1), 'human_train_loss =', '{:.9f}'.format(human_loss_train))
    print('Epoch:', '%04d' % (epoch + 1), 'dog_train_loss =', '{:.9f}'.format(dog_loss_train))
    
    print('Epoch:', '%04d' % (epoch + 1), 'human_test_loss =', '{:.9f}'.format(human_loss_test))
    print('Epoch:', '%04d' % (epoch + 1), 'dog_test_loss =', '{:.9f}'.format(dog_loss_test))
    

    error_human_train[epoch] = human_loss_train
    error_dog_train[epoch]   = dog_loss_train
    error_human_test[epoch]  = human_loss_test
    error_dog_test[epoch]    = dog_loss_test
    error_human_train.tofile("./result/model/error_human_train.bin")
    error_human_test.tofile("./result/model/error_human_test.bin")
    error_dog_train.tofile("./result/model/error_dog_train.bin")
    error_dog_test.tofile("./result/model/error_dog_test.bin")
    
    save_path = saver.save(sess, "./result/model/model.ckpt")    
    PFNN.save_network((sess.run(P0_dog.alpha),sess.run(P1.alpha),sess.run(P2_dog.alpha)),
                      (sess.run(P0_dog.beta), sess.run(P1.beta), sess.run(P2_dog.beta)),
                      50,
                      './result/dogNN'
                      )
    PFNN.save_network((sess.run(tf.concat([P0_dog.getAlpha(index_inputTrajectory,-1),P0_human.alpha, P0_dog.getAlpha(index_terrain,-1)],axis = -1)),
                       sess.run(P1.alpha),
                       sess.run(tf.concat([P2_human.alpha[:,:8,:], P2_dog.getAlpha(index_outputTrajectory,1),P2_human.alpha[:,8:,:]],axis = 1))),
                      (sess.run(P0_human.beta), sess.run(P1.beta), sess.run(tf.concat([P2_human.beta[...,8:],P2_dog.getBeta(index_outputTrajectory,-1), P2_human.beta[...,8:]], axis = -1))),
                      50,
                      './result/humanNN'
                      )
    
    if epoch>0 and epoch%5==0:
        path_dog   = './result/dogweights/dogNN%03i' % epoch
        path_human = './result/humanweights/humanNN%03i' % epoch
        if not os.path.exists(path_dog):
            os.makedirs(path_dog)
            
        PFNN.save_network((sess.run(P0_dog.alpha),sess.run(P1.alpha),sess.run(P2_dog.alpha)),
                          (sess.run(P0_dog.beta), sess.run(P1.beta), sess.run(P2_dog.beta)),
                          50,
                          path_dog
                         )
        if not os.path.exists(path_human):
            os.makedirs(path_human)
            
        PFNN.save_network((sess.run(tf.concat([P0_dog.getAlpha(index_inputTrajectory,-1),P0_human.alpha, P0_dog.getAlpha(index_terrain,-1)],axis = -1)),
                           sess.run(P1.alpha),
                           sess.run(tf.concat([P2_human.alpha[:,:8,:], P2_dog.getAlpha(index_outputTrajectory,1),P2_human.alpha[:,8:,:]],axis = 1))
                           ),
                          (sess.run(P0_human.beta), sess.run(P1.beta), 
                           sess.run(tf.concat([P2_human.beta[...,8:],P2_dog.getBeta(index_outputTrajectory,-1), P2_human.beta[...,8:]], axis = -1))
                           ),
                          50,
                          path_human
                         )        
print('Learning Finished!')
#-----------------------------above is model training----------------------------------