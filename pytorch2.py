
# coding: utf-8

# In[*]

import torch
from torch.autograd import Variable


# In[*]

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)


# In[*]

print(tensor)
print(variable)


# In[*]

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
print(t_out)
print(v_out)


# In[*]

v_out.backward() # backpropagation from v_out
print(variable.grad)


# In[*]

print(variable)


# In[*]

print(variable.data)


# In[*]

print(variable.data.numpy())

