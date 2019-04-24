import torch
import torch.nn as nn

from libpy import Init             #Import my own functions for test

class NLRWDense(nn.Module):
    def __init__(self, input_features, output_features, work_style = "NL", distant_parameter = 0.1, device = "cuda"):
        super(NLRWDense, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.work_style = work_style
        self.distant_parameter = distant_parameter
        self.device = device

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-1, 1)


    def forward(self, input):
        Zero = torch.zeros(1).to(self.device)
        print(self.weight)
        wei2 = torch.sum(torch.mul(self.weight, self.weight), dim = 1)
        inp2 = torch.sum(torch.mul(input, input), dim = 1)
        #print(input, inp2)
        wei2 = wei2.reshape([1, -1])
        inp2 = inp2.reshape([-1, 1])
        #print(wei2, inp2)

        Tul = torch.exp( 
                        -self.distant_parameter * 
                            torch.sqrt( 
                                torch.max(
                                    inp2 - 2 * torch.mm(input, self.weight.t()) + wei2, 
                                    Zero
                                )
                            )
                        )

        #print(Tul)
        #Main Train and call function
        if self.work_style == "NL":
            SumTul = torch.sum(Tul, dim = 1)

            SumTul = SumTul.reshape([-1, 1])
            outputs = torch.div(Tul, SumTul)
            #print(outputs)
            return outputs
      
        elif self.work_style == "RW":
            I = torch.eye(input.size()[0]).to(self.device)

            Tuu = torch.exp(
                            - self.distant_parameter * 
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

            return outputs

        else:
            ValueError("The input method must be 'NL' for Non-Linear cluster classification or 'RW' for Random Walk Classification")
            return


    def extra_repr(self):
        #Output the io size for visible
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features is not None
        )





