from Modules.Neural_Net import Net
import torch
import matplotlib.pyplot as plt

'We start  by making the function we want to approximate'

x = torch.linspace(-1, 1, 99).reshape(-1, 1)
y = torch.sin(2 * 3.14 * x**2) #This is the target function we want to approximate, change it with whatever you want

'Set up the neural network with losses and optimizer'

net  =  Net(torch.cos, 1, 10, 10) # this is a NN with 1 input and a two hidden layers wiht 10 neurons each
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = .01)

'Training the net'
loss_check = 1
k=0
while loss_check > .01:
    k +=1 
    optimizer.zero_grad()
    y_pred = net(x)
    loss = loss_func(y_pred, y)
    loss.backward()
    optimizer.step()
    loss_check = loss.item()
    if k %10 ==0:
        print(f' The loss is: {loss.item()}')
    if k > 300:
        print(f'failed to converge after {k} iterations with loss {loss.item()}')
        break

'Plotting the results'
plt.plot(x.detach().numpy(), y.detach().numpy(), label='Target Function')
plt.plot(x.detach().numpy(), y_pred.detach().numpy(), label='NN Approximation')
plt.legend()
plt.title('1D Function Approximation with Neural Network')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()



