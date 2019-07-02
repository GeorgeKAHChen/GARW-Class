#=============================================================================
#
#       Group Attribute Random Walk Program
#       NLRWClass.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This is the Class file we build for our new random walk classification
#       You can use this class directly. The parameter usage is same as linear
#       Classification
#
#=============================================================================
import torch
import torch.nn as nn

class NLRWDense(nn.Module):
    def __init__(self, 
                input_features, 
                output_features, 
                work_style = "NL", 
                UL_distant = 1, 
                UU_distant = 1, 
                device = "cuda"):

        super(NLRWDense, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.work_style = work_style
        self.UU_distant = UU_distant
        self.UL_distant = UL_distant
        self.device = device

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-1, 1)


    def Output(self):
        import numpy as np
        Arr = ""
        weight = self.weight.cpu().detach().numpy()
        #print(weight)
        for i in range(0, len(weight)):
            for j in range(0, len(weight[i])):
                Arr += str(weight[i][j])
                Arr += "\t"
            Arr += "\n"
        FileName = "Output/RWModel" + str(self.input_features) + ".out"
        File = open(FileName, "w")
        File.write(Arr)
        File.close()
        return


    def forward(self, input):
        if len(input) == 1:
            NLRWDense.Output(self)
            return

        Zero = torch.zeros(1).to(self.device)
        epsilon = torch.Tensor([1e-5]).to(self.device)
        #print(self.weight)
        wei2 = torch.sum(torch.mul(self.weight, self.weight), dim = 1)
        inp2 = torch.sum(torch.mul(input, input), dim = 1)
        #print(input, inp2)
        wei2 = wei2.reshape([1, -1])
        inp2 = inp2.reshape([-1, 1])
        #print(wei2, inp2)

        Tul = torch.exp( 
                        -self.UL_distant * 
                            torch.sqrt( 
                                torch.max(
                                    inp2 - 2 * torch.mm(input, self.weight.t()) + wei2, 
                                    Zero
                                )
                            )
                        )

        #print(Tul)
        #Main Train and call function
        if self.work_style == "NL" or len(input) == 1:
            SumTul = torch.sum(Tul, dim = 1)

            SumTul = SumTul.reshape([-1, 1])
            SumTul = torch.max(SumTul, epsilon)
            outputs = torch.div(Tul, SumTul)
            #print(outputs)
            return outputs
      
        elif self.work_style == "RW":
            I = torch.eye(input.size()[0]).to(self.device)

            Tuu = torch.exp(
                            - self.UU_distant * 
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

            SumMatrix = torch.max(SumMatrix, epsilon)

            Pul = Tul / SumMatrix
            Puu = Tuu / SumMatrix

            IPuu = I - Puu
            try:
                IPuu = torch.inverse(IPuu)
                outputs = torch.mm(IPuu, Pul)
                outputs = torch.max(outputs, epsilon)
            except:
                outputs = Pul
                outputs = torch.max(outputs, epsilon)
            return outputs

        else:
            ValueError("The input method must be 'NL' for Non-Linear cluster classification or 'RW' for Random Walk Classification")
            return


    def extra_repr(self):
        #Output the io size for visible
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )















class GMMDense(nn.Module):
    def __init__(self, 
                input_features, 
                output_features,  
                device = "cuda"):

        super(GMMDense, self).__init__()
        self.input_features  = input_features
        self.output_features = output_features
        self.device          = device

        self.prob            = nn.Parameter(torch.Tensor(output_features, 1)).to(self.device)
        self.Mu              = nn.Parameter(torch.Tensor(output_features, input_features)).to(self.device)
        self.Sigma           = nn.Parameter(torch.Tensor(output_features, input_features, input_features)).to(self.device)
        
        self.prob.data.uniform_(0, 1)
        self.Mu.data.uniform_(0, 1)
        self.Sigma.data.uniform_(0.5, 1)


    def Unit_prob(self):
        Zero      = torch.zeros(self.output_features, 1).to(self.device)
        self.prob = nn.Parameter(torch.max(self.prob, Zero))
        sum_prob  = torch.sum(self.prob).to(self.device)
        self.prob = nn.Parameter(self.prob / sum_prob)


    def Output_Model(self):
        import numpy as np
        inp_fea = self.input_features
        oup_fea = self.output_features
        prob    = self.prob.cpu().detach().numpy()
        Mu      = self.Mu.cpu().detach().numpy()
        Sigma   = self.Sigma.cpu().detach().numpy()
        OupArr  = str(inp_fea) + "\t" + str(oup_fea) + "\n"

        for i in range(0, len(prob)):
            OupArr += str(prob[i][1])
            OupArr += "\n"
            for j in range(0, len(Mu[i])):
                OupArr += str(Mu[i][j])
                OupArr += "\t"
            OupArr += "\n"
            for j in range(0, len(Sigma[i])):
                for k in range(0, len(Sigma[i][j])):
                    OupArr += str(Sigma[i][j][k])
                    OupArr += "\t"
                OupArr += "\n"
        
        FileName = "Output/Model" + str(inp_fea) + ".out"
        File = open(FileName, "w")
        File.write(OupArr)
        File.close()
        print("Model Saved Succeed, file name: Model.out")
        return



    def Sigma_Cal(self):
        Zero = torch.zeros(1).to(self.device)
        epsilon = torch.Tensor([1e-5]).to(self.device)

        sigma_inv = torch.zeros([self.output_features, self.input_features, self.input_features]).to(self.device)
        sigma_det = torch.zeros(self.output_features).to(self.device)
        for i in range(0, len(self.Sigma)):
            sigma_inv[i] = torch.inverse(self.Sigma[i])

            det = torch.sqrt(torch.max(torch.det(self.Sigma[i]), Zero)).to(self.device)
            if torch.gt(det, epsilon):
                det = torch.Tensor(1).to(self.device)
                sigma_det[i] = 0.3989422804014327/det
            else:
                sigma_det[i] = 0

        return sigma_inv, sigma_det


    def forward(self, input):
        import math
        Zero = torch.zeros(1).to(self.device)
                                            # def para =: 0 as tensor
        epsilon = torch.Tensor([1e-5]).to(self.device)
        GMMDense.Unit_prob(self)            # Initial self.prob parameter as normal
        
        if len(input) == 1:
            GMMDense.Output_Model(self)     # Save Model
            return 

        sigma_inv, sigma_det = GMMDense.Sigma_Cal(self)
                                            # cal sigma-inverse and sigma-det to calculate
        outputs = torch.zeros(self.output_features, len(input)).to(self.device)

        for j in range(0, self.output_features):
            x_neg_mu = input - self.Mu[j]
            #outputs[j] = sigma_det[j] * torch.exp(-0.5 * (
            outputs[j] = torch.exp(-0.5 * (
                torch.max( torch.sum(torch.mul(torch.mm(x_neg_mu, sigma_inv[j]), x_neg_mu), dim = 1), Zero)
                ))
        outputs = outputs.t()
        SumOps = torch.max(torch.sum(outputs, dim = 1), epsilon)
        SumOps = torch.reshape(SumOps, [-1, 1])
        outputs /= SumOps
        #print(outputs)
        return outputs



    def extra_repr(self):
        #Output the io size for visible
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )
        