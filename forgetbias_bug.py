
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
import numpy as np
import coremltools
from coremltools.models.neural_network import datatypes, NeuralNetworkBuilder

weights = np.array([[.1, .1, .1, .2, .2, .2, .3,.3, .3, .4, .4, .4],[.5,.5, .5,.6,.6,.6,.7,.7,.7,.8,.8,.8],
[.9,.9,.9,-.1,.9,.9, -.2,.9,.9, -.3, .9,.9],[-.4,.9,.9,-.5,.9,.9,-.6,.9,.9,-.7,.9,.9]])

forward_file = 'rnn1_1.mlmodel'
reverse_file = 'rnn1_2.mlmodel'
lr_scope ='lr/SeqLstm'
rl_scope = 'rl/SeqLstm'   
lr_kernel = lr_scope + "/rnn/basic_lstm_cell/kernel:0"    
rl_kernel = rl_scope + "/rnn/basic_lstm_cell/kernel:0"
lr_bias = lr_scope + "/rnn/basic_lstm_cell/bias:0"
rl_bias = rl_scope + "/rnn/basic_lstm_cell/bias:0"
num_outputs = 3
input_shape=(1,2,1) # batch, sequence, features
tf_input_shape=(2,1,1) # sequence, batch, features

input_tensor_name = 'data'
output_tensor_name = 'rnnoutput'

run_with_bug = False

def tf_model(sess, input_shape, num_outputs, lr_scope,
             rl_scope):  
    pholder = tf.placeholder(tf.float32, shape=input_shape, name='rnninput')
    with tf.variable_scope(lr_scope):
        lstm_cell = rnn_cell.BasicLSTMCell(num_outputs)

        net1, state1 = rnn.dynamic_rnn(
            lstm_cell, pholder, time_major=True, dtype=pholder.dtype)

    with tf.variable_scope(rl_scope):
        input = array_ops.reverse_v2(pholder, [0])
        lstm_cell = rnn_cell.BasicLSTMCell(num_outputs)
        net, state = rnn.dynamic_rnn(
            lstm_cell, input, time_major=True, dtype=input.dtype)
        net = array_ops.reverse_v2(net, [0])
        
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
        
    return pholder, net1, net, state1, state


def gen_coreml(val_W_x, val_W_h, val_b, val_W_x_back, val_W_h_back, val_b_back, input_shape, num_outputs, coreml_file1, coreml_file2):
    input_features = [(input_tensor_name, datatypes.Array(input_shape[-1]))]
    output_features = [(output_tensor_name, datatypes.Array(input_shape[1] * num_outputs))]
    builder = NeuralNetworkBuilder(input_features, output_features)
    builder.add_unilstm('horizlstm', val_W_h, val_W_x, val_b, 
                          num_outputs, input_shape[-1], 
                          [input_tensor_name, "h_input", "c_input"],
                          [output_tensor_name, "h_output", "c_output"],
                          forget_bias=run_with_bug) 
    optional_inputs=[]
    optional_inputs.append(("h_input", num_outputs)) 
    optional_inputs.append(("c_input", num_outputs))
    optional_outputs=[]
    optional_outputs.append(("h_output", num_outputs)) 
    optional_outputs.append(("c_output", num_outputs))
    builder.add_optionals(optional_inputs, optional_outputs)
    coremltools.utils.save_spec(builder.spec, coreml_file1)
    
    builder = NeuralNetworkBuilder(input_features, output_features)
    builder.add_unilstm('horizlstm', 
                          val_W_h_back, val_W_x_back, val_b_back, 
                          num_outputs, input_shape[-1], 
                          [input_tensor_name, "h_input", "c_input"],
                          [output_tensor_name, "h_output", "c_output"],
                          forget_bias=run_with_bug) 
    optional_inputs=[]
    optional_inputs.append(("h_input", num_outputs)) 
    optional_inputs.append(("c_input", num_outputs))
    optional_outputs=[]
    optional_outputs.append(("h_output", num_outputs)) 
    optional_outputs.append(("c_output", num_outputs))
    builder.add_optionals(optional_inputs, optional_outputs)
    coremltools.utils.save_spec(builder.spec, coreml_file2)
    
def getGateWeights(allWeights):
    input, cell, forget, output = array_ops.split(value=allWeights, num_or_size_splits=4, axis=0)
    return input,cell,forget,output

def getGateBiases(allBiases):
    input, cell, forget, output = array_ops.split(value=allBiases, num_or_size_splits=4)
    return input,cell,forget,output
    
def getLSTMWeights(var, input_shape, num_outputs):
    input_weights, recursion_weights = array_ops.split(value=var, num_or_size_splits=[input_shape[-1], num_outputs], axis=0)
    input, cell, forget, output = getGateWeights(tf.transpose(input_weights))
    input_recurse, cell_recurse, forget_recurse, output_recurse = getGateWeights(tf.transpose(recursion_weights))
    return input, cell, forget, output, input_recurse, cell_recurse, forget_recurse, output_recurse
    
def varForName(name):
    var = [v for v in tf.trainable_variables() if v.name == name][0]
    return var

def getLSTMBiases(var):
    input, cell, forget, output = getGateBiases(var)
    return input, cell, forget, output
    
def get_tfmodel(lr_kernel, rl_kernel, sess, tf_inputshape, num_outputs, lr_scope, rl_scope):
    pholder, net1, net, state1, state = tf_model(sess, tf_inputshape, num_outputs, lr_scope, rl_scope)

    # get/assign forward weights (in real life, read in a checkpoint)
    w_f = varForName(lr_kernel)
    assign_op=w_f.assign(weights)
    sess.run(assign_op)

    # get/assign reverse weights
    w_b = varForName(rl_kernel)
    assign_op=w_b.assign(weights)
    sess.run(assign_op)
    return w_f, w_b, pholder, net1, net, state1, state
    
def generate(lr_kernel, rl_kernel, lr_bias, rl_bias, input_shape, tf_inputshape, num_outputs, coreml_file1, coreml_file2,
             lr_scope, rl_scope):
    with tf.Session(graph=tf.Graph()) as sess:
        w_f, w_b,_,_,_,_,_ = get_tfmodel(lr_kernel, rl_kernel, sess, tf_inputshape, num_outputs, lr_scope, rl_scope)
            
        # figure out where the weight matrices are
        input, cell, forget, output, input_recurse, cell_recurse, forget_recurse, output_recurse = getLSTMWeights(w_f, tf_inputshape, num_outputs)
        input_back, cell_back, forget_back, output_back, input_recurse_back, cell_recurse_back, forget_recurse_back, output_recurse_back = getLSTMWeights(w_b, tf_inputshape, num_outputs)
        
        # figure out where the biases are
        b_var = varForName(lr_bias)
        b_i, b_c, b_f, b_o = getLSTMBiases(b_var)
        bb_var = varForName(rl_bias)
        bb_i, bb_c, bb_f, bb_o = getLSTMBiases(bb_var)
        
        # get the actual values for weights/biases
        important_results = [input, cell, forget, output, input_recurse, cell_recurse, forget_recurse, output_recurse, 
                             input_back, cell_back, forget_back, output_back, input_recurse_back, cell_recurse_back, 
                             forget_recurse_back, output_recurse_back, b_i, b_c, b_f, b_o, bb_i, bb_c, bb_f, bb_o
                             ]
        val_i, val_c, val_f, val_o, val_ir, val_cr, val_fr, val_or, val_ib, val_cb, val_fb, val_ob, val_irb, val_crb, val_frb, val_orb,val_bi, val_bc, val_bf, val_bo, val_bbi, val_bbc, val_bbf, val_bbo = sess.run(important_results)
      
        # put the weights/biases in the format coreml wants
        W_h, W_x, W_h_back, W_x_back, b, b_back = ([], [], [], [], [], [])
        W_h.append(val_ir)
        W_h.append(val_fr)
        W_h.append(val_or)
        W_h.append(val_cr)
        
        W_x.append(val_i)
        W_x.append(val_f)
        W_x.append(val_o)
        W_x.append(val_c)
        
        W_h_back.append(val_irb)
        W_h_back.append(val_frb)
        W_h_back.append(val_orb)
        W_h_back.append(val_crb)
        
        W_x_back.append(val_ib)
        W_x_back.append(val_fb)
        W_x_back.append(val_ob)
        W_x_back.append(val_cb)
        
        b.append(val_bi)
        
        if not run_with_bug:
            # override val_bf to force "forget_bias"
            val_bf = val_bf + np.ones((num_outputs))
        b.append(val_bf)
        
        b.append(val_bo)
        b.append(val_bc)
        
        b_back.append(val_bbi)
        
        
        if not run_with_bug:
            # override val_bbf to force "forget_bias"
            val_bbf = val_bbf + np.ones((num_outputs))
        
        b_back.append(val_bbf)
        b_back.append(val_bbo)
        b_back.append(val_bbc)
        gen_coreml(W_x, W_h, b, W_x_back, W_h_back, 
            b_back,input_shape, num_outputs, coreml_file1, coreml_file2)
    
def run_tf(inputarray, inputshape):
    with tf.Session(graph=tf.Graph()) as sess:
        _,_, pholder, net1, net, state1, state = get_tfmodel(lr_kernel, rl_kernel, sess, inputshape, num_outputs, lr_scope, rl_scope)
        
        result = sess.run([net1, net, state1, state], feed_dict={pholder:inputarray})
        return result

   
def run_coreml(inputarray, lastresult, coreml_file):
    model =  coremltools.models.MLModel(coreml_file)
    input = {input_tensor_name: inputarray}
    if not lastresult is None:
        input['h_input'] = lastresult['h_output']
        input['c_input'] = lastresult['c_output']
    
    predictions = model.predict(input)
    return predictions


# generate the coreml file.
generate(lr_kernel, rl_kernel, lr_bias, rl_bias, input_shape, tf_input_shape, num_outputs, forward_file, reverse_file, lr_scope, rl_scope)
       
tf_input_shape = (2, 1, 1)
tf_inputarray = np.ones(tf_input_shape)
inputarray = np.transpose(tf_inputarray, (1, 0, 2))
tf_result = run_tf(tf_inputarray, tf_input_shape) 
print('tf timestamp 1:' + str(tf_result[0][1]))

coreml_result = None
for j in range(input_shape[1]):
    cmlinput = inputarray[0][j]
    coreml_result = run_coreml(cmlinput, coreml_result, forward_file)
    if j == 1:
        print('Coreml timestamp 1' + str(j) + ":" + str(coreml_result['rnnoutput']))

    