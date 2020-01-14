'''
Python 3.6.8 |Anaconda custom (64-bit)| (default, Feb 21 2019, 18:30:04) [MSC v.1916 64 bit (AMD64)] on win32
runfile('F:/_academic/ongoing(aca)/s1#pvldb2017/实验代码/01=USG/pytorch1.py', wdir='F:/_academic/ongoing(aca)/s1#pvldb2017/实验代码/01=USG')
1.0.1
True
cpu 8.640682935714722 tensor(1000286.5000)
cuda:0 0.45053720474243164 tensor(1000286.5625, device='cuda:0')
cuda:0 0.010990142822265625 tensor(1000286.5625, device='cuda:0')

'''

import torch
import time

print(torch.__version__)
print(torch.cuda.is_available())

a = torch.randn(10000,10000)
b = torch.randn(10000,10000)

t0 = time.time()
c  = torch.matmul(a,b)
t1 = time.time()

print(a.device,t1-t0,c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

t0 = time.time()
c  = torch.matmul(a,b)
t1 = time.time()

print(a.device,t1-t0,c.norm(2))

t0 = time.time()
c  = torch.matmul(a,b)
t1 = time.time()

print(a.device,t1-t0,c.norm(2))