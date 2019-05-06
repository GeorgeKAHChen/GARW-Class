#=============================================================================
#
#       Group Attribute Random Walk Program
#       cpu_vs_gpu.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This is a file which can confident that random walk with gpu(cuda) 
#       is faster than cpu. Also, this is the file which can determine the 
#       random walk layer are not wrong.
#       You can get detail usage in README.md file
#
#=============================================================================


import torch
import time
from sklearn.datasets import load_iris
import numpy as np
from copy import deepcopy

def forward(weight, input, UL_distant, UU_distant, device = "cuda"):
    weight = weight.to(device)
    input = input.to(device)
    Zero = torch.zeros(1).to(device)
    #print(self.weight)
    wei2 = torch.sum(torch.mul(weight, weight), dim = 1)
    inp2 = torch.sum(torch.mul(input, input), dim = 1)
    #print(input, inp2)
    wei2 = wei2.reshape([1, -1])
    inp2 = inp2.reshape([-1, 1])
    #print(wei2, inp2)

    Tul = torch.exp(-UL_distant *
                        torch.sqrt(
                            torch.max(
                                inp2 - 2 * torch.mm(input, weight.t()) + wei2,
                                Zero
                            )
                        )
                    )

    I = torch.eye(input.size()[0]).to(device)

    Tuu = torch.exp(- UU_distant *
                        torch.sqrt(
                            torch.max(
                                inp2 - 2*torch.mm(input, input.t()) + inp2.t(),
                                Zero
                            )
                        )
                    )
    Tuu = Tuu - I

    SumTuu = torch.sum(Tuu, dim = 1)
    SumTuu = torch.reshape(SumTuu, [-1, 1])
    #print(inp2)
    SumTul = torch.sum(Tul, dim = 1)
    SumTul = torch.reshape(SumTul, [-1, 1])

    SumMatrix = SumTul + SumTuu
    SumMatrix = torch.reshape(SumMatrix, [-1, 1])

    Pul = Tul / SumMatrix
    Puu = Tuu / SumMatrix

    outputs = torch.mm(torch.inverse(I - Puu), Pul)
    #Init.ArrOutput(outputs)
    return outputs


if __name__ == '__main__':
    iris = load_iris()
    Xold = np.array(iris.data[:, :4])
    X = deepcopy(Xold)
    Zold = [Xold[0], Xold[50], Xold[100]]
    Z = torch.Tensor(Zold)

    for i in range(0, 100):
        X = torch.Tensor(X)  
        start = time.time()
        forward(Z, X, 0.5, 0.5, device = "cuda")
        end = time.time()
        t1 = end - start

        start = time.time()
        forward(Z, X, 0.5, 0.5, device = "cpu")
        end = time.time()
        t2 = end - start

        print("epoch = ", i+1, "cuda time: ", t1, ", cpu time: ", t2)
        New = []
        for j in range(0, i + 2):
            for k in range(0, len(Xold)):
                New.append(Xold[k])
        X = deepcopy(New)


        