import tensorflow as tf
import numpy as np 
#Forward propagation
# input >weight>hidden layer 1(activation  funtion) > weights > hidden 2(Activation fun)
# > weights> output layers '''

#BAck propagation
#campare output to intended output >cost or loss function
#optimization function >minimize cost(Adamoptimizer,SGD<adaGrad)

from POS_NEG import create_feature_sets_and_labels
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

nclasses = 2
batch_size = 100

x = tf.placeholder('float',[None, len(train_x[0])])
y = tf.placeholder('float')

def neuralnetwork(data):
    #(inputdata * weights) + biases 
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} 


    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}   

    output_layers = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,nclasses])),
                       'biases':tf.Variable(tf.random_normal([nclasses]))}

    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases'])

    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layers['weights']) + output_layers['biases']

    return output

def train_neural_network(x):
    prediction = neuralnetwork(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    #learing rat = 0.01
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #feed forward + back propagation = 1 epochs
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        #training data  optimizing weights
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i<len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
           

                _,c = sess.run([optimizer, cost], feed_dict = {x: batch_x,y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        
        
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)