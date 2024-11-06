#!/usr/bin/env python

# Code used to evaluate the 3 optimizers for one neuron 

import random
import numpy as np
import operator
import matplotlib.pyplot as plt


seed = 0           
random.seed(seed)
np.random.seed(seed)

from ComputationalGraphPrimer import *

class NewComputationalGraphPrimer (ComputationalGraphPrimer):
    #We override this method for the plot
    def run_training_loop_one_neuron_model(self, training_data):
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.

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
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

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


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record_1 = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)     ##  FORWARD PROP of data
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])  ##  Find loss
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record_1.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))
            self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids)  ## BACKPROP loss
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()   
        return loss_running_record_1

class SGDPlusComputationalGraphPrimer (ComputationalGraphPrimer):
    ##MODIFIED CODE---------------------------------------------
    #Initialize new atributes
    def __init__(self, u=0, *args, **kwargs):
        self.u = u
        super().__init__(*args, **kwargs)
    ##END MODIFIED CODE---------------------------------------------

    def run_training_loop_one_neuron_model(self, training_data):
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.
        ##MODIFIED CODE---------------------------------------------
        #We initialize auxiliar variables used to update parameters          
        self.v_t = {param: 0 for param in self.learnable_params}
        self.bias_v_t = 0
        ##END MODIFIED CODE-----------------------------------------
        
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
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

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


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record_2 = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)     ##  FORWARD PROP of data
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])  ##  Find loss
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record_2.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))
            self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids)  ## BACKPROP loss
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()
        return loss_running_record_2
        
    def backprop_and_update_params_one_neuron_model(self, data_tuples_in_batch, predictions, y_errors_in_batch, deriv_sigmoids):
        """
        This function implements the equations shown on Slide 61 of my Week 3 presentation in our DL 
        class at Purdue.  All four parameters defined above are lists of what was either supplied to the
        forward prop function or calculated by it for each training data sample in a batch.
        """
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]                  ## These two statements align the
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}   ##   the input vars 
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            ## For each param, sum the partials from every training data sample in batch
            partial_of_loss_wrt_param = 0.0
            for j in range(self.batch_size):
                vals_for_input_vars_dict =  dict(zip(input_vars, list(data_tuples_in_batch[j])))
                partial_of_loss_wrt_param   +=   -  y_errors_in_batch[j] * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoids[j]
            partial_of_loss_wrt_param /=  float(self.batch_size)

            ##MODIFIED CODE---------------------------------------------
            #Apply SGD+ formulas
            #NESTEROV
            v_t_plus_1 = self.u * self.v_t[param] + partial_of_loss_wrt_param
            step = self.learning_rate * v_t_plus_1
            self.vals_for_learnable_params[param] -= step
            self.v_t[param] = v_t_plus_1
        
        partial_of_loss_wrt_param=0.0
        for j in range(self.batch_size):
            #We keep the consistency of using a negative value for the partial_of_loss_wrt_param as given in the initial code for the weights
            partial_of_loss_wrt_param   +=   -  y_errors_in_batch[j] * deriv_sigmoids[j]
        partial_of_loss_wrt_param /=  float(self.batch_size)
        
        #Apply SGD+ formulas
        #NESTEROV
        bias_v_t_plus_1 = self.u * self.bias_v_t + partial_of_loss_wrt_param
        step_bias = self.learning_rate * bias_v_t_plus_1
        self.bias -= step_bias
        self.bias_v_t = bias_v_t_plus_1
        ##END MODIFIED CODE---------------------------------------------
        
class AdamComputationalGraphPrimer (ComputationalGraphPrimer):
    ##MODIFIED CODE---------------------------------------------
    #Initialize new atributes
    def __init__(self, epsilon, beta1=0, beta2=0, *args, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)
    ##END MODIFIED CODE---------------------------------------------

        
    def backprop_and_update_params_one_neuron_model(self, data_tuples_in_batch, predictions, y_errors_in_batch, deriv_sigmoids, t):
        """
        This function implements the equations shown on Slide 61 of my Week 3 presentation in our DL 
        class at Purdue.  All four parameters defined above are lists of what was either supplied to the
        forward prop function or calculated by it for each training data sample in a batch.
        """
        input_vars = self.independent_vars
        input_vars_to_param_map = self.var_to_var_param[self.output_vars[0]]                  ## These two statements align the
        param_to_vars_map = {param : var for var, param in input_vars_to_param_map.items()}   ##   the input vars 
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            ## For each param, sum the partials from every training data sample in batch
            partial_of_loss_wrt_param = 0.0
            for j in range(self.batch_size):
                vals_for_input_vars_dict =  dict(zip(input_vars, list(data_tuples_in_batch[j])))
                partial_of_loss_wrt_param   +=   -  y_errors_in_batch[j] * vals_for_input_vars_dict[param_to_vars_map[param]] * deriv_sigmoids[j]
            partial_of_loss_wrt_param /=  float(self.batch_size)
            
            ##MODIFIED CODE---------------------------------------------
            #Apply Adam formulas
            m_t_plus_1 = self.beta1 * self.m_t[param] + (1 - self.beta1) * partial_of_loss_wrt_param
            v_t_plus_1 = self.beta2 * self.v_t[param] + (1 - self.beta2) * (partial_of_loss_wrt_param**2)
            
            m_t_plus_1_hat = m_t_plus_1 / (1 - self.beta1**t)
            v_t_plus_1_hat = v_t_plus_1 / (1 - self.beta2**t)
            
            step = self.learning_rate * (m_t_plus_1_hat / np.sqrt(v_t_plus_1_hat + self.epsilon))
            self.vals_for_learnable_params[param] -= step
            
            self.v_t[param] = v_t_plus_1
            self.m_t[param] = m_t_plus_1
        
        partial_of_loss_wrt_param=0.0
        for j in range(self.batch_size):
            #We keep the consistency of using a negative value for the partial_of_loss_wrt_param as given in the initial code for the weights
            partial_of_loss_wrt_param   +=   -  y_errors_in_batch[j] * deriv_sigmoids[j]
        partial_of_loss_wrt_param /=  float(self.batch_size)
        
        #Apply Adam formulas
        bias_m_t_plus_1 = self.beta1 * self.bias_m_t + (1 - self.beta1) * (partial_of_loss_wrt_param)
        bias_v_t_plus_1 = self.beta2 * self.bias_v_t + (1 - self.beta2) * (partial_of_loss_wrt_param**2)
        
        bias_m_t_plus_1_hat = bias_m_t_plus_1 / (1 - self.beta1**t)
        bias_v_t_plus_1_hat = bias_v_t_plus_1 / (1 - self.beta2**t)
        
        bias_step = self.learning_rate * (bias_m_t_plus_1_hat / np.sqrt(bias_v_t_plus_1_hat + self.epsilon))
        
        self.bias -= bias_step
        
        self.bias_v_t = bias_v_t_plus_1
        self.bias_m_t = bias_m_t_plus_1
        ##END MODIFIED CODE---------------------------------------------
        
    def run_training_loop_one_neuron_model(self, training_data):
        """
        The training loop must first initialize the learnable parameters.  Remember, these are the 
        symbolic names in your input expressions for the neural layer that do not begin with the 
        letter 'x'.  In this case, we are initializing with random numbers from a uniform distribution 
        over the interval (0,1).
        """
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}

        self.bias = random.uniform(0,1)                   ## Adding the bias improves class discrimination.
                                                          ##   We initialize it to a random number.
        ##MODIFIED CODE---------------------------------------------
        #We initialize auxiliar variables used to update parameters
        self.m_t = {param: 0 for param in self.learnable_params}
        self.v_t = {param: 0 for param in self.learnable_params}
        self.bias_m_t = 0
        self.bias_v_t = 0
        ##END MODIFIED CODE-----------------------------------------

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
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]   ## Associate label 0 with each sample
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]   ## Associate label 1 with each sample

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


        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_iterations = 0.0                                    ##  Average the loss over iterations for printing out 
                                                                           ##    every N iterations during the training loop.
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples_in_batch = data[0]
            class_labels_in_batch = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples_in_batch)     ##  FORWARD PROP of data
            loss = sum([(abs(class_labels_in_batch[i] - y_preds[i]))**2 for i in range(len(class_labels_in_batch))])  ##  Find loss
            avg_loss_over_iterations += loss / float(len(class_labels_in_batch))
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_iterations))                 ## Display average loss
                avg_loss_over_iterations = 0.0                                                     ## Re-initialize avg loss
            y_errors_in_batch = list(map(operator.sub, class_labels_in_batch, y_preds))
            
            ### CODE MODIFIED----------------------------------------------------
            #We pass the iteration useful to update the parameters with Adam
            self.backprop_and_update_params_one_neuron_model(data_tuples_in_batch, y_preds, y_errors_in_batch, deriv_sigmoids, t = i+1)  ## BACKPROP loss
            ### END CODE MODIFIED----------------------------------------------------
        
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()  
        return loss_running_record 

#Run the training using the 3 optimizers and plotting the results
cgp = NewComputationalGraphPrimer(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )    

cgp.parse_expressions()
training_data_cgp = cgp.gen_training_data()
SGD_loss = cgp.run_training_loop_one_neuron_model( training_data_cgp )


cgp_SGDPlus = SGDPlusComputationalGraphPrimer(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
               u = 0.9
      )

cgp_SGDPlus.parse_expressions()
training_data_cgp_SGDPlus = cgp_SGDPlus.gen_training_data()
SGD_plus_loss = cgp_SGDPlus.run_training_loop_one_neuron_model( training_data_cgp_SGDPlus )

cgp_Adam = AdamComputationalGraphPrimer(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
               beta1 = 0.9,
               beta2 = 0.99,
               epsilon = 1e-8
      )

cgp_Adam.parse_expressions()
training_data_cgp_Adam = cgp_Adam.gen_training_data()
SGD_adam_loss = cgp_Adam.run_training_loop_one_neuron_model( training_data_cgp_Adam )


plt.figure()
plt.plot(SGD_loss)
plt.plot(SGD_plus_loss)
plt.plot(SGD_adam_loss)
plt.legend(["SGD","SGD+","Adam"])
plt.title("Figure 1: One Neuron learning rate = 1e-2")
plt.show()