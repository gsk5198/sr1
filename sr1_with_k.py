import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# In the first time, do the download job;
download_flag = True 

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=download_flag,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=download_flag,
    transform=ToTensor()
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
    
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f"Using {device} device")


import torch.nn.functional as F # 引用神经网络常用函数包，不具有可学习的参数




#--------------------------------------------------------------------
# Model 1: CNN, Trainable=431080 (0.43M，说明CNN比MLP效果好很多！)
# sgdm: Epochs=30, each epoch 36.005 sec., The best test error 11.40%
#------------------------------------------------------------------
# model 1: Trainable=431080
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # Conv2d参数：1表示输入通道，20表示输出通道，5表示conv核大小，1表示步长stride
        # input [N, C_in, H_in, W_in]; output [N,C_out,H_out,W_out]
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10) 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    
    
    
    
#--------------------------------------------------------------------
# Model 2: CNN, Trainable=21840 (参数特别少，仅21K) 
#sgdm: Epochs=30, each epoch 18.338 sec., The best test error 12.69% 
#--------------------------------------------------------------------
# model 2: Trainable=21840
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()        
        self.CNN = nn.Sequential(
           nn.Conv2d(1, 10, kernel_size=5),
           nn.MaxPool2d(2),
           nn.ReLU(),
           nn.Dropout2d(),
           nn.Conv2d(10, 20, kernel_size=5),
           nn.MaxPool2d(2),
           nn.ReLU(),
           nn.Flatten(),
           nn.Linear(320, 50),
           nn.ReLU(),
           nn.Linear(50, 10)
           # nn.Softmax()     # we use cross-entropy that will do softmax
         )
    def forward(self, x):
        logits = self.CNN(x)
        return logits

    


import numpy as np



def show_model_size(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total = 0
    Trainable = 0
    NonTrainable = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total += mulValue  # 总参数量
        if param.requires_grad:
            Trainable += mulValue  # 可训练参数量
        else:
            NonTrainable += mulValue  # 非可训练参数量
    
    # show information
    print(f'Parameters: Total={Total}, Trainable={Trainable}, Non-trainable={NonTrainable}')  
 


# construct the network
model = Net1().to(device)
#model = Net2().to(device)
print(model)
show_model_size(model)
    


from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt




# In the first time, do the download job;
download_flag = True 

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=download_flag,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=download_flag,
    transform=ToTensor()
)



batch_size = 64 # the number of examples in a mini-batch
class_number = 10 # FashionMNIST data has 10 classes




# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
    
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f"Using {device} device")




# Define model
# Model 1: 784->512->512->10, 迭代40个epochs只能勉强到80%准确率。
# Model 2: 784->1000->500->200->10，参考另一个帖子，多一层，而且第1隐层1000个节点
# Model 3: 我这里对model2改了下，用1024, 512，256。看结果怎样。
# Epoch 40: 540 sec; 
#   training: Examples 60000, Accuracy 82.54%, Avg loss 0.504440
#   test: Examples 10000, Accuracy 81.27%, Avg loss 0.531622
#--------------------------------------------------------------------
# Model 4: 784->512->256->128->64->10, Trainable=575050（0.57M, 我觉得这个模型不错！）
# (1)training: Examples 60000, Accuracy 89.64%, Avg loss 0.285150
#    test: Examples 10000, Accuracy 86.15%, Avg loss 0.417517
# (2)training: Examples 60000, Accuracy 90.23%, Avg loss 0.263889
#    test: Examples 10000, Accuracy 86.99%, Avg loss 0.394932
# (3)test: Examples 10000, Accuracy 84.66%, Avg loss 0.438142; Epochs=20
# (4)test: Examples 10000, Accuracy 86.54%, Avg loss 0.394373, epochs=30
# (5)test: Examples 10000, Accuracy 87.02%, Avg loss 0.373832, epochs=30, MultiStepLR
# (6)test: Examples 10000, Accuracy 85.95%, Avg loss 0.396527, epochs=30, MultiStepLR
# (7)test, sgdm: each epoch: 10.761 seconds, The best test error is 17.19%
# (8)test, sgd(不加冲量)：each epoch: 10.257 seconds, The best test error is 14.98%
#--------------------------------------------------------------------
# Model 5: 784->512->256->128->64->32->10, 
#    test: Examples 10000, Accuracy 85.46%, Avg loss 0.416489
#    test: Examples 10000, Accuracy 86.57%, Avg loss 0.424272
# Model 6: 784->512->512->256->128->64->32->10, 
#    test: Examples 10000, Accuracy 86.25%, Avg loss 0.415294
# Model 7: 784->1024->512->256->128->64->10, Trainable: 1501770 (1.5M)
#    test: Examples 10000, Accuracy 86.07%, Avg loss 0.393048
#--------------------------------------------------------------------
# Model 8: 784->64->10, ten epochs, 65%->74%, each epoch 7 sec.
#          784->64->10, ten epochs, 58%->72%, each epoch 7 sec.
#--------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.eta = 1e-3 # initial learning rate 
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.Sigmoid(), 
            nn.Tanh(),
            nn.Linear(64, 64), 
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(64, 64), 
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(64, 32), 
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(32, 32),
            #nn.ReLU(),
            #nn.Sigmoid(), 
            nn.Tanh(),
            nn.Linear(32, 32), 
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(32, 10) 
        )
        
        # ten layers: 
        #sgd: Epochs=10, each epoch 9.209 sec., The best test error 90.00% (无法训练)
        #sgd: Epochs=10, each epoch 9.190 sec., The best test error 90.00% (无法训练)
        #sgd: Epochs=10, each epoch 8.877 sec., The best test error 90.00% (无法训练)         
        #sgdm:Epochs=10, each epoch 9.262 sec., The best test error 90.00% (无法训练)
        #sgdm:Epochs=10, each epoch 9.185 sec., The best test error 90.00% (无法训练)
        #sgdm:Epochs=10, each epoch 8.959 sec., The best test error 90.00% (无法训练)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 32), 
            nn.ReLU(),
            nn.Linear(32, 32), 
            nn.ReLU(),
            nn.Linear(32, 10) 
        )
        
        # 结论：十层 Sigmoid 网络就无法bp训练了。
        # sgd: Epochs=10, each epoch 10.384 sec., The best test error 90.00%
        # sgdm: Epochs=10, each epoch 10.577 sec., The best test error 90.00%
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64), 
            nn.Sigmoid(),
            nn.Linear(64, 64), 
            nn.Sigmoid(),
            nn.Linear(64, 64), 
            nn.Sigmoid(),
            nn.Linear(64, 32), 
            nn.Sigmoid(),
            nn.Linear(32, 32), 
            nn.Sigmoid(),
            nn.Linear(32, 32), 
            nn.Sigmoid(),
            nn.Linear(32, 10) 
        )
        
        #sgd: each epoch: 10.455 seconds, The best test error is 69.04%
        #sgd: Epochs=10, each epoch 11.299 sec., The best test error 63.62%
        #sgd: Epochs=10, each epoch 11.056 sec., The best test error 59.61%
        #sgd: Epochs=10, each epoch 10.780 sec., The best test error 59.06%
        #sgdm: each epoch: 10.581 seconds, The best test error is 18.66%
        #sgdm: each epoch: 11.392 seconds, The best test error is 17.63%
        #sgdm: each epoch: 11.408 seconds, The best test error is 17.13%
        #self.linear_relu_stack = nn.Sequential(
        #    nn.Linear(28*28, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, 64), 
        #    nn.ReLU(),
        #    nn.Linear(64, 10) 
        #)
        
        
        #sgd: each epoch: 6.944 seconds, The best test error is 26.45%
        #sgdm: each epoch: 7.003 seconds, The best test error is 16.35%
        #self.linear_relu_stack = nn.Sequential(
        #    nn.Linear(28*28, 64),
        #    nn.Tanh(),
        #    nn.Linear(64, 10)
        #)
        
        
    def setLearningRate(self, t):
        if t>0 and t%5==0:
            self.eta *= 0.5
        
        
        

    def forward(self, x):
        x = self.flatten(x)       # to get a 784-dim vector
        # print(x.size()) # torch.Size([64, 784])
        #logits = self.linear_relu_stack(x)    #test error 90%
        #logits = self.linear_sigmoid_stack(x) #test error 90%
        logits = self.linear_tanh_stack(x)      #test error ~16%
        return logits
    
    
    

import numpy as np
def show_model_size(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total = 0
    Trainable = 0
    NonTrainable = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total += mulValue  # 总参数量
        if param.requires_grad:
            Trainable += mulValue  # 可训练参数量
        else:
            NonTrainable += mulValue  # 非可训练参数量
    
    # show information
    print(f'Parameters: Total={Total}, Trainable={Trainable}, Non-trainable={NonTrainable}')  

    
    
'''
    
# construct the network
model = NeuralNetwork().to(device)
print(model)
show_model_size(model)

'''
import math
import torch
from torch.optim.optimizer import Optimizer
from tabulate import tabulate
from torch import Tensor
from typing import List
version_higher = ( torch.__version__ >= "1.5.0" )

'''
def twoloop(lam, u, grad): #计算矩阵H和向量grad的乘积（其中lam和u分别为系数lambda和向量u的列表）

    n = len(lam) #向量序列的长度
    if n == 0 :
        return grad
    else:
        q = grad * 1
        for i in range(n):
            q.add(lam[i]*torch.mul(u[i],torch.mul(u[i],grad)))
    return q

def find_the_oldest(lst,alpha=0.9):
    a=lst[0]
    k=0
    for i in range(len(lst)):
        if lst[i]<a:
            k=i
        lst[i] *= alpha
    return k

def sr1(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         maximize: bool,
         alpha:float):

    global S, Y, RHO,RHO_alpha,pre_grad,RHO2,pre_data
    
    
    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
       
        step = state_steps[i]


        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

                 
        if len(S)<=i:
            S.append([])
            Y.append([])
            RHO.append([])
            RHO_alpha.append([])
            RHO2.append([])
            pre_grad.append(None)
            pre_data.append(None)
        s=S[i]
        y=Y[i]
        rho=RHO[i]
        rho2=RHO2[i]
        rho_alpha=RHO2[i]
        if pre_grad[i] != None:
            yk = grad - pre_grad[i]
            sk = param.data - pre_data[i]
        else:
            yk = grad
            sk = param.data
        y.append(yk)
        s.append(sk)
        Hy = twoloop(rho,rho2,yk)
        syk = sk-Hy
        b = torch.mul(syk,yk).sum()
        a = torch.mul(syk,syk).sum()
        #print(sk)
        c = torch.norm(syk,p='fro')
        ##a = torch.norm(syk,p=2,dim=0)
      
        rho.append(a/b)
        rho_alpha.append(a/b)
        rho2.append(syk/c)
        global NUM_of_u,total_num,big_num
        try:
            NUM_of_u==0
            total_num==0
            big_num==0
        except:
            NUM_of_u , total_num , big_num = 25 , 0 , 0
        
        #print(a,b)
        
        
        
        total_num+=1
        #print(len(rho))
        if len(rho)> NUM_of_u: #弃掉最旧向量
            a=rho[0]
            k=0
            #print(len(rho))
            for j in range(len(rho)):
                #print(a,j,rho[j])
                if rho[j]<a:
                    k=j
            #k=0
            
            #print(k)
            if k >= 0.8 * NUM_of_u: big_num += 1
            rho_alpha.pop(k)
            rho.pop(k)
            rho2.pop(k)'''
'''
            
        
        if len(rho)>0 and len(RHO[0]) >= NUM_of_u: #弃掉最旧向量
            k = find_the_oldest(rho_alpha)
            rho.pop(k)
            rho_alpha.pop(k)
            rho2.pop(k)
'''
'''
        pk = alpha*twoloop(rho,rho2,grad)
        pre_grad[i] = grad
        pre_data[i] = param.data
    
       
        S[i] = s
        Y[i] = y
        RHO[i] = rho

        
        param.data.add_(pk, alpha=-lr)
class SR1(Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,warmup=500,init_lr=None, 
                 weight_decay=0, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,warmup=warmup,init_lr=init_lr, 
                        weight_decay=weight_decay,base_lr=lr, amsgrad=amsgrad, maximize=maximize)
        super(SR1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SR1, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Calculate current lr
                    if state['step'] < group['warmup']:
                        curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmup'] + group['init_lr']
                    else:
                        curr_lr = group['lr']
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            sr1(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=curr_lr,
                   weight_decay=group['weight_decay'],
                   eps=group['eps'],
                   maximize=group['maximize'],
                   alpha=0.05)
                   
                
         
            
        return loss
'''



def twoloop(lam, u, grad): #计算矩阵H和向量grad的乘积（其中lam和u分别为系数lambda和向量u的列表）

    n = len(lam) #向量序列的长度
    if n == 0 :
        return grad
    else:
        q = grad * 1
        for i in range(n):
            q.add(lam[i]*torch.mul(u[i],torch.mul(u[i],grad)))
    return q

def find_the_oldest(lst,alpha=0.9):
    a=lst[0]
    k=0
    for i in range(len(lst)):
        if lst[i]<a:
            k=i
        lst[i] *= alpha
    return k

def sr1(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         maximize: bool,
         alpha:float):

    global S, Y, RHO,RHO_alpha,pre_grad,RHO2,pre_data
    
    
    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]


        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

                 
        if len(S)<=i:
            #S.append([])
            #Y.append([])
            RHO.append([])
            RHO_alpha.append([])
            RHO2.append([])
            pre_grad.append(None)
            pre_data.append(None)
        #s=S[i]
        #y=Y[i]
        rho=RHO[i]
        rho2=RHO2[i]
        #rho_alpha=RHO2[i]
        if pre_grad[i] != None:
            yk = grad - pre_grad[i]
            sk = param.data - pre_data[i]
        else:
            yk = grad
            sk = param.data
        
        
        #y.append(yk)
        #s.append(sk)
        
        
        Hy = twoloop(rho,rho2,yk)
        syk = sk-Hy
        b = torch.mul(syk,yk).sum()
        a = torch.mul(syk,syk).sum()
        #print(sk)
        c = torch.norm(syk,p='fro')
        ##a = torch.norm(syk,p=2,dim=0)
      
        rho.append(a/b)
        #rho_alpha.append(a/b)
        rho2.append(syk/c)
        global NUM_of_u,total_num,big_num
        try: #检测是否有控制u的数量的全局变量NUM_of_u
            NUM_of_u==0
            total_num==0
            big_num==0
        except:
            NUM_of_u , total_num , big_num = 25 , 0 , 0
        
        #print(a,b)
        
        
        
        total_num+=1
        #print(len(rho))
        if len(rho)> NUM_of_u: #弃掉最旧向量
            a=rho[0]
            k=0
            '''
            #print(len(rho))
            for j in range(len(rho)):
                #print(a,j,rho[j])
                if rho[j]<a:
                    k=j
            #k=0
            
            #print(k)
            if k >= 0.8 * NUM_of_u: big_num += 1  #记录被踢出的u在列表中的大致位置
            #rho_alpha.pop(k)'''
            rho.pop(k)
            rho2.pop(k)
        '''
            
        
        if len(rho)>0 and len(RHO[0]) >= NUM_of_u: #弃掉最旧向量
            k = find_the_oldest(rho_alpha)
            rho.pop(k)
            rho_alpha.pop(k)
            rho2.pop(k)
        '''
    
        pk = alpha*twoloop(rho,rho2,grad)
        pre_grad[i] = grad
        pre_data[i] = param.data
    
        '''
        S[i] = s
        Y[i] = y
        RHO[i] = rho
        RHO2[i] = rho2
        '''
        
        param.data.add_(pk, alpha=-lr)
class SR1(Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,warmup=500,init_lr=None, 
                 weight_decay=0, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,warmup=warmup,init_lr=init_lr, 
                        weight_decay=weight_decay,base_lr=lr, amsgrad=amsgrad, maximize=maximize)
        super(SR1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SR1, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Calculate current lr
                    if state['step'] < group['warmup']:
                        curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmup'] + group['init_lr']
                    else:
                        curr_lr = group['lr']
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            sr1(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=curr_lr,
                   weight_decay=group['weight_decay'],
                   eps=group['eps'],
                   maximize=group['maximize'],
                   alpha=0.05)
                   
                
         
            
        return loss

#it contains the softmax operation, so don't use softmax
loss_fn = nn.CrossEntropyLoss() 



# 加入冲量后效果提升明显；若不加冲量，可能效果很差！
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) 
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
lr, num_epochos = 0.001, 100
'''
#optimizer = SR1(model.parameters(),lr=1, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, amsgrad=False)
optimizer = torch.optim.Adam(model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)'''

# 多步学习率，改进不是很大
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

# Here define a function to train the model
def my_train(dataloader, model, loss_fn, optimizer):
    show_flag = False
    size = len(dataloader.dataset)
    model.train()  # set the train mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # print(y.size()) # torch.Size([64])

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Show progress
        if show_flag and (batch % 200 == 0):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
# the original test function; I change it for both training data and test data
def my_eval(dataloader, model, loss_fn, data_string):
    size = len(dataloader.dataset)
    if size<1:
        print("eval error: size < 1")
        return 1.0
    num_batches = len(dataloader)
    model.eval()  # set the evaluation mode
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #print(f"X.size={X.size()}, y.size={y.size()}")
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss /= num_batches
    accuracy = correct / size
    err = 1.0 - accuracy
    print(f"{data_string}: Examples {size}, Error {(100*err):>0.2f}%, Avg loss {avg_loss:>6f}")
    return err,avg_loss

#-------------------------------------------------------------    
# classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
#           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]    
#-------------------------------------------------------------
# Show classification results of some example
def show_examples():
    classes = {
        0: "0:T-shirt",  #T-shirt/top
        1: "1:Trouser",
        2: "2:Pullover",
        3: "3:Dress",
        4: "4:Coat",
        5: "5:Sandal",
        6: "6:Shirt",
        7: "7:Sneaker",
        8: "8:Bag",
        9: "9:Ankle Boot",
        }
    # load the best model with the highest accuracy
    model.load_state_dict(torch.load("model02.pth"))
    print("Load PyTorch Model State from model02.pth")
    model.eval()  # set the evaluation mode
    softmax = nn.Softmax(dim=1)
    figure = plt.figure(figsize=(12, 12))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_data), size=(1,)).item()
        img, label = test_data[sample_idx]
        # print(img.shape)     #torch.Size([1, 28, 28])
        # note that nn.Conv2d requires input as 
        #      x[ batch_size, channels, height_1, width_1 ], otherwize error
        img2 = img.unsqueeze(0)
        pred = model(img2)
        pred2 = softmax(pred) #torch.Size([1, 10])
        #print(f'pred2: {pred2.size()}, {pred2}')
        actual = classes[label]
        predicted = classes[int( pred2[0].argmax(0) )]
        #print(f'Predicted: "{predicted}", Actual: "{actual}"')     
        figure.add_subplot(rows, cols, i)
        plt.title(f'{actual}->{predicted}')
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    
    
    


# the main entry
import time
start = time.time()
test_err_min = 1.0
epochs = 30     #20


file=open('data_tanh_try.txt','w')
#NUM_of_u = 10
for i in range(4):
    
    S = []
    Y = []
    RHO = []
    pre_grad = []
    RHO2 = []
    pre_data = []
    RHO_alpha = []
    
    #NUM_of_u+=5
    
    
    error_lst=[]
    loss_lst=[]
    model = NeuralNetwork().to(device)
    #model = Net1().to(device)
    #model = Net2().to(device)
    print(model)
    #print(NUM_of_u)
    show_model_size(model)
    optimizer = SR1(model.parameters(),lr=1, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False)
    start = time.time()
    for t in range(epochs):
        
        total_num=0
        big_num=0
        
        elapsed_seconds = time.time() - start
        print(f"--Epoch {t+1}: --{elapsed_seconds:>.3f} sec.")
        my_train(train_dataloader, model, loss_fn, optimizer)
        train_err,loss=my_eval(train_dataloader, model, loss_fn, "training")
        test_err,loss=my_eval(test_dataloader, model, loss_fn, "test")
        #file.write(str(test_err))
        if test_err_min > test_err:
            test_err_min = test_err
            # to serialize the internal state dictionary (containing the model parameters)
            torch.save(model.state_dict(), "model02.pth")
            print(f"test_err={(100*test_err):>.2f}%, Save PyTorch Model State to model02.pth")
        error_lst.append(test_err)
        loss_lst.append(loss)
        print(total_num,big_num)
    elapsed_seconds = time.time() - start
    file.write(str(error_lst))
    file.write("\n")
    file.write(str(loss_lst))
    file.write("\n")
    print(f'\nEpochs={epochs}, each epoch {elapsed_seconds/epochs:>.3f} sec., The best test error {(100*test_err_min):>.2f}%')
    #show_examples()
print('end')
file.close()