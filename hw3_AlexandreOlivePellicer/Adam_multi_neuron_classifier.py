#!/usr/bin/env python

# Code used to evaluate the 9 configurations of the parameteters beta1 and beta2 of the Adam optimizer
import random
import numpy as np
import operator
import matplotlib.pyplot as plt
import time

seed = 100           
random.seed(seed)
np.random.seed(seed)

from ComputationalGraphPrimer import *

class Adam_Multi_ComputationalGraphPrimer(ComputationalGraphPrimer):
      def __init__(self, beta1=0, beta2=0, *args, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        super().__init__(*args, **kwargs)
  
      def backprop_and_update_params_multi_neuron_model(self, predictions, y_errors, t):
        """
        First note that loop index variable 'back_layer_index' starts with the index of
        the last layer.  For the 3-layer example shown for 'forward', back_layer_index
        starts with a value of 2, its next value is 1, and that's it.

        In the code below, the outermost loop is over the data samples in a batch. As shown
        on Slide 73 of my Week 3 lecture, in order to calculate the partials of Loss with
        respect to the learnable params, we need to backprop the prediction errors and 
        the gradients of the Sigmoid.  For the purpose of satisfying the requirements of
        SGD, the backprop of the prediction errors and the gradients needs to be carried
        out separately for each training data sample in a batch.  That's what the outer
        loop is for.

        After we exit the outermost loop, we average over the results obtained from each
        training data sample in a batch.

        Pay attention to the variable 'vars_in_layer'.  These store the node variables in
        the current layer during backpropagation.  
        """
        ## Eq. (24) on Slide 73 of my Week 3 lecture says we need to store backproped errors in each layer leading up to the last:
        pred_err_backproped_at_layers =   [ {i : [None for j in range( self.layers_config[i] ) ]  
                                                                  for i in range(self.num_layers)} for _ in range(self.batch_size) ]
        ## This will store "\delta L / \delta w" you see at the LHS of the equations on Slide 73:
        partial_of_loss_wrt_params = {param : 0.0 for param in self.all_params}
        ## For estimating the changes to the bias to be made on the basis of the derivatives of the Sigmoids:
        bias_changes =   {i : [0.0 for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        for b in range(self.batch_size):
            pred_err_backproped_at_layers[b][self.num_layers - 1] = [ y_errors[b] ]
            for back_layer_index in reversed(range(1,self.num_layers)):             ## For the 3-layer network, the first val for back_layer_index is 2 for the 3rd layer
                input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]     ## This is a list of 8 two-element lists  --- since we have two nodes in the 2nd layer
                deriv_sigmoids =  self.gradient_vals_for_layers[back_layer_index]   ## This is a list eight one-element lists, one for each batch element
                vars_in_layer  =  self.layer_vars[back_layer_index]                 ## A list like ['xo']
                vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   ## A list like ['xw', 'xz']
                vals_for_input_vars_dict =  dict(zip(vars_in_next_layer_back, self.forw_prop_vals_at_layers[back_layer_index - 1][b]))   
                ## For the next statement, note that layer_params are stored in a dict like        
                ##       {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
                ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
                layer_params = self.layer_params[back_layer_index]         
                transposed_layer_params = list(zip(*layer_params))                  ## Creating a transpose of the link matrix, See Eq. 30 on Slide 77
                for k,var1 in enumerate(vars_in_next_layer_back):
                    for j,var2 in enumerate(vars_in_layer):
                        pred_err_backproped_at_layers[b][back_layer_index - 1][k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]]
                                                                                       * pred_err_backproped_at_layers[b][back_layer_index][i]
                                                                                                                  for i in range(len(vars_in_layer))])
                for j,var in enumerate(vars_in_layer):
                    layer_params = self.layer_params[back_layer_index][j]           ##  ['cp', 'cq']   for the end layer
                    input_vars_to_param_map = self.var_to_var_param[var]            ## These two statements align the    {'xw': 'cp', 'xz': 'cq'}
                    param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}   ##   and the input vars   {'cp': 'xw', 'cq': 'xz'}

                    ##  Update the partials of Loss wrt to the learnable parameters between the current layer
                    ##  and the previous layer. You are accumulating these partials over the different training
                    ##  data samples in the batch being processed.  For each training data sample, the formula
                    ##  being used is shown in Eq. (29) on Slide 77 of my Week 3 slides:
                    for i,param in enumerate(layer_params):
                        partial_of_loss_wrt_params[param]   +=   pred_err_backproped_at_layers[b][back_layer_index][j] * \
                                                                        vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoids[b][j]
                ##  We will now estimate the change in the bias that needs to be made at each node in the previous layer
                ##  from the derivatives the sigmoid at the nodes in the current layer and the prediction error as
                ##  backproped to the previous layer nodes:
                for k,var1 in enumerate(vars_in_next_layer_back):
                    for j,var2 in enumerate(vars_in_layer):
                        if back_layer_index-1 > 0:
                            bias_changes[back_layer_index-1][k] += pred_err_backproped_at_layers[b][back_layer_index - 1][k] * deriv_sigmoids[b][j] 
 
        ## Now update the learnable parameters.  The loop shown below carries out SGD mandated averaging
        for param in partial_of_loss_wrt_params: 
            partial_of_loss_wrt_param = - partial_of_loss_wrt_params[param] /  float(self.batch_size)   
            # step = self.learning_rate * partial_of_loss_wrt_param 
            # self.vals_for_learnable_params[param] += step
            
            m_t_plus_1 = self.beta1 * self.m_t[param] + (1 - self.beta1) * partial_of_loss_wrt_param
            v_t_plus_1 = self.beta2 * self.v_t[param] + (1 - self.beta2) * (partial_of_loss_wrt_param**2)
            
            m_t_plus_1_hat = m_t_plus_1 / (1 - self.beta1**t)
            v_t_plus_1_hat = v_t_plus_1 / (1 - self.beta2**t)
            
            step = self.learning_rate * (m_t_plus_1_hat / np.sqrt(v_t_plus_1_hat + self.epsilon))
            self.vals_for_learnable_params[param] -= step
            
            self.v_t[param] = v_t_plus_1
            self.m_t[param] = m_t_plus_1
                       

        ##  Finally we update the biases at all the nodes that aggregate data:      
        for layer_index in range(1,self.num_layers):           
            for k in range(self.layers_config[layer_index]):
                #self.bias[layer_index][k]  +=  self.learning_rate * ( bias_changes[layer_index][k] / float(self.batch_size) )
                partial_of_loss_wrt_param = - ( bias_changes[layer_index][k] / float(self.batch_size) )
                bias_m_t_plus_1 = self.beta1 * self.bias_m_t[layer_index][k] + (1 - self.beta1) * (partial_of_loss_wrt_param)
                bias_v_t_plus_1 = self.beta2 * self.bias_v_t[layer_index][k] + (1 - self.beta2) * (partial_of_loss_wrt_param**2)
                
                bias_m_t_plus_1_hat = bias_m_t_plus_1 / (1 - self.beta1**t)
                bias_v_t_plus_1_hat = bias_v_t_plus_1 / (1 - self.beta2**t)
                
                bias_step = self.learning_rate * (bias_m_t_plus_1_hat / np.sqrt(bias_v_t_plus_1_hat + self.epsilon))
                
                self.bias[layer_index][k] -= bias_step
                
                self.bias_v_t[layer_index][k] = bias_v_t_plus_1
                self.bias_m_t[layer_index][k] = bias_m_t_plus_1
                
                
                
      def run_training_loop_multi_neuron_model(self, training_data):
        
        class DataLoader:
            """
            To understand the logic of the dataloader, it would help if you first understand how 
            the training dataset is created.  Search for the following function in this file:

                             gen_training_data(self)
           
            As you will see in the implementation code for this method, the training dataset
            consists of a Python dict with two keys, 0 and 1, the former points to a list of 
            all Class 0 samples and the latter to a list of all Class 1 samples.  In each list,
            the data samples are drawn from a multi-dimensional Gaussian distribution.  The two
            classes have different means and variances.  The dimensionality of each data sample
            is set by the number of nodes in the input layer of the neural network.

            The data loader's job is to construct a batch of samples drawn randomly from the two
            lists mentioned above.  And it mush also associate the class label with each sample
            separately.
            """
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]    ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]    ## Associate label 1 with each sample

            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])

            def _getitem(self):    
                cointoss = random.choice([0,1])                            ## When a batch is created by getbatch(), we want the
                                                                           ##   samples to be chosen randomly from the two lists
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            

            def getbatch(self):
                batch_data,batch_labels = [],[]                            ## First list for samples, the second for labels
                maxval = 0.0                                               ## For approximate batch data normalization
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]          ## Normalize batch data       
                batch = [batch_data, batch_labels]
                return batch                

        ##  The training loop must first initialize the learnable parameters.  Remember, these are the 
        ##  symbolic names in your input expressions for the neural layer that do not begin with the 
        ##  letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        ##  over the interval (0,1):
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        ##  In the same  manner, we must also initialize the biases at each node that aggregates forward
        ##  propagating data:
        self.bias =   {i : [random.uniform(0,1) for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                          ##  Average the loss over iterations for printing out 
                                                                                ##    every N iterations during the training loop.
                                                                                
        self.m_t = {param: 0 for param in self.learnable_params}
        self.v_t = {param: 0 for param in self.learnable_params}
        self.bias_m_t = {i : [0 for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        self.bias_v_t = {i : [0 for j in range( self.layers_config[i] ) ]  for i in range(1, self.num_layers)}
        
        min_loss = 1000
           
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                       ## FORW PROP works by side-effect 
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]           ## Predictions from FORW PROP
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]       ## Get numeric vals for predictions
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])  ## Calculate loss for batch
            #We save the value of the minimum loss
            if loss<min_loss:
                  min_loss = loss
            loss_avg = loss / float(len(class_labels))                                              ## Average the loss over batch
            avg_loss_over_iterations += loss_avg                                                    ## Add to Average loss over iterations
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                  ## Display avg loss
                avg_loss_over_iterations = 0.0                                                      ## Re-initialize avg-over-iterations loss
            y_errors_in_batch = list(map(operator.sub, class_labels, y_preds))
            ##MODIFIED CODE---------------------------------------------
            self.backprop_and_update_params_multi_neuron_model(y_preds, y_errors_in_batch, t=i+1)
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()  
        return loss_running_record, loss, min_loss
            ##END MODIFIED CODE-----------------------------------------  


beta1 = [0.8, 0.95, 0.99]
beta2 = [0.89, 0.9, 0.95]

loss = []
final_loss = []
minimum_loss = []
time_ = []
beta1_ = []
beta2_ = []


for b1 in beta1:
      for b2 in beta2:
        start_time = time.time()
        cgp = Adam_Multi_ComputationalGraphPrimer(
                      num_layers = 3,
                      layers_config = [4,2,1],                         # num of nodes in each layer
                      expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                                      'xz=bp*xp+bq*xq+br*xr+bs*xs',
                                      'xo=cp*xw+cq*xz'],
                      output_vars = ['xo'],
                      dataset_size = 5000,
                      learning_rate = 1e-3,
                      training_iterations = 40000,
                      batch_size = 8,
                      display_loss_how_often = 100,
                      debug = True,
                      beta1 = b1,
                      beta2 = b2,
              )

        cgp.parse_multi_layer_expressions()

        training_data = cgp.gen_training_data()

        loss_running_record, last_loss, min_loss = cgp.run_training_loop_multi_neuron_model( training_data )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        loss.append(loss_running_record)
        final_loss.append(last_loss)
        minimum_loss.append(min_loss)
        time_.append(elapsed_time)
        beta1_.append(b1)
        beta2_.append(b2)

legend = []        
plt.figure()
for i, _ in enumerate(beta1_):
  plt.plot(loss[i])
  leg= f"beta1: {beta1_[i]}, beta2: {beta2_[i]}"
  legend.append(leg)
plt.legend(legend)
plt.title("Figure : Adam Optimizer under 9 configurations")
plt.show()

for i, _ in enumerate(beta1_):
  print(f"beta1: {beta1_[i]}, beta2: {beta2_[i]}, final loss: {final_loss[i]}, minimum loss: {minimum_loss[i]}, time: {time_[i]} sec")

