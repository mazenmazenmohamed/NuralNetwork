import numpy as np
def segmoid(x) :
    return (1.0 / (1+np.exp(-x)))
def segmoid_deravatives (x):
    return x*(1-x)
class nuralnetworks :
    def __init__(self,x,y):
        self.input=x
        self.weights1=np.random.rand(self.input.shape[1],4)
        self.weights2=np.random.rand(4,1)
        self.y=y
        self.output=np.zeros(self.y.shape)
        
    def feedforward(self):    
         self.layer1=segmoid(np.dot(x,self.weights1))
         self.output= segmoid(np.dot(self.layer1,self.weights2))
    def backprob (self):
        dweights2=np.dot(self.layer1.T,(2*(self.y-self.output)*segmoid_deravatives(self.output)))
        dweights1=np.dot(self.input.T,  (np.dot(2*(self.y-self.output)*segmoid_deravatives(self.output),self.weights2.T)*segmoid_deravatives(self.layer1)))
        self.weights1+= dweights1
        self.weights2+=dweights2
        
x=np.array([[0,0,1],
           [0,1,1],
           [1,0,1],
           [1,1,1]])    
y=np.array([[0],
            [1],
            [1],
            [0]])   
NN=nuralnetworks(x,y)
for i in range(78000):
    NN.feedforward()
    NN.backprob()
print(NN.output)    
print(y)    
print(np.round(NN.output))