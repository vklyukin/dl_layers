Softmax layer
---
Your task is to write a [softmax layer](https://pytorch.org/docs/master/generated/torch.nn.Softmax.html) with NumPy. It 
is a function that takes as input a vector z of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers.
You should implement class `Softmax`, which is an inheritor of an abstract class `Module`.  
The interface is inspired by [PyTorch library](https://pytorch.org/).  

> If you are not sure about your understanding of the backpropagation, we recommend that you watch this [video](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

## Details
1. input: **`batch_size x n_feats`**
2. output: **`batch_size x n_feats`**
3. $\text{softmax}(x)_i = \frac{\exp x_i} {\sum_j \exp x_j}$
4. Recall that  softmax(𝑥)==softmax(𝑥−const) . It makes possible to avoid computing exp() from large argument.
5. Implement method `updateOutput`, which is called in `forward` method of base class. So this method 
just computes the output using the input. 
6. Implement method `updateGradInput`. You should implement it to pass to the previous layer in your neural network. 
For single layer it is not necessary, but obligatorily in more complex networks.
7. We do not need to calculate the gradients with respect to the layer's parameteres because we have not ones.


If you are confused with code signature here is the code for the `Module` class:
```python
class Module(object):
    """
    Basically, you can think of a utils as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = utils.forward(input)
    
    The utils should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        gradInput = utils.backward(input, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the utils.
        """
        return self.updateOutput(input)

    def backward(self,input, gradOutput):
        """
        Performs a backpropagation step through the utils, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput
    

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.
        
        Make sure to both store the data in `output` field and return it. 
        """
        
        # The easiest case:
            
        # self.output = input 
        # return self.output
        
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the utils with respect to its own input. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.
        
        The shape of `gradInput` is always the same as the shape of `input`.
        
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        
        # The easiest case:
        
        # self.gradInput = gradOutput 
        # return self.gradInput
        
        pass   
    
    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the utils with respect to its own parameters.
        No need to override if utils has no parameters (e.g. ReLU).
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Zeroes `gradParams` variable if the utils has params.
        """
        pass
        
    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the utils does not have parameters return empty list. 
        """
        return []
        
    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the utils does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the utils.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the utils.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every utils if you want 
        to have readable description. 
        """
        return "Module"
```