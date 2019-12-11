# Logistic Regression

# Importing the libraries
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def_get_data(pares_de_datos):
    mitad_de_pares = pares_de_datos//2
    dimen = 2
    return np.random.randn(pares_de_datos, dimen)*3, mitad_de_pares, dimen


def_grafics(labelx, labely, color, s, alpha):
    if plt.scatter(labelx, labely, c=color, s=s, alpha=alpha):
        print('Grafics [OK]')
    else:
        print('Grafics [FAIL]')

def_grafics_with_plot(labelx, labely):
    if plt.plot(labelx, labely):
        print('Grafics [OK]')
    else:
        print('Grafics [FAIL]')


deftraining_losses(inpt, outpt, model, loss_function, optimizer, losses, iterations):
    for i in range(iterations):
        result = model(inpt)
        loss = loss_function(result, outpt)
        losses.append(loss.data)

        optimizer.zero_grad()
        loss.backward()                                                                  # Back Propagation
        optimizer.step()
    print('last loss: {}.'.format(float(loss)))
    _grafics_with_plot(range(iterations), losses)
    return model



defcreate_model(inpt, outpt):
    model = nn.Sequential(nn.Linear(2,1), nn.Sigmoid())
    loss_function = nn.BCELoss() # BCE es el Binary cross entropy
    optimizer = optim.SGD(model.parameters(), lr=0.015)
    model = training_losses(inpt, outpt, model, loss_function, optimizer, losses=[], iterations=2000)
    return model


defmain(data, mitad_de_pares, pares_de_datos, colors):
    target = np.array([0]*mitad_de_pares + [1]*mitad_de_pares).reshape(pares_de_datos,1)
    inpt = torch.from_numpy(data).float().requires_grad_()
    outpt = torch.from_numpy(target).float()
    model = create_model(inpt, outpt)
    input_data = torch.Tensor([[-5, -6]])
    prediction = model(input_data).data[0][0] > 0.5
    print('La probabilidad sera de: {}, por lo tanto la es de color: {}.'.format(prediction.data, colors[prediction]))

if __name__ == '__main__':
    pares_de_datos = 100
    data, mitad_de_pares, dimen = _get_data(pares_de_datos)
    colors = ['blue', 'red']
    color = np.array([colors[0]]*mitad_de_pares + [colors[1]]*mitad_de_pares).reshape(pares_de_datos)
    _grafics(data[:,0], data[:,1], color, s=75, alpha=0.6)
    data[:mitad_de_pares, :] -= 3*np.ones((mitad_de_pares, dimen))
    data[mitad_de_pares:, :] += 3*np.ones((mitad_de_pares, dimen))
    _grafics(data[:,0], data[:,1], color, s=75, alpha=0.6)
    main(data, mitad_de_pares, pares_de_datos, colors)