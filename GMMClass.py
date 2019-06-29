#=============================================================================
#
#       Group Attribute Random Walk Program
#       GMMClass.py
#
#       Copyright by KazukiAmakawa, all right reserved
#
#=======================================================
#
#       Intro
#       This is the Class file we build for our new GMM Classification
#
#=============================================================================
import torch
import torch.nn as nn

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
        OupArr = str(self.input_features) + "\t" + str(self.output_features) + "\n"
        for i in range(0, len(self.prob)):
            print(i)
            OupArr += str(self.prob[i])
            OupArr += "\n"
            for j in range(0, len(self.Mu[i])):
                OupArr += str(self.Mu[i][j])
                OupArr += "\t"
            OupArr += "\n"
            for j in range(0, len(self.Sigma[i])):
                for k in range(0, len(self.Sigma[i][j])):
                    OupArr += str(self.Sigma[i][j][k])
                    OupArr += "\t"
                OupArr += "\n"
        final_arr = ""
        for i in range(0, len(OupArr)):
            if not (OupArr[i] == "." or OupArr[i] == "\n" or OupArr[i] == "\t" or (ord(OupArr[i]) >= ord("0") and ord(OupArr[i]) <= ord("9"))):
                continue
            else:
                final_arr += OupArr[i]
        import os
        os.system("rm -rf Model.out")
        FileName = "Model.out"
        File = open(FileName, "w")
        File.write(final_arr)
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
        GMMDense.Unit_prob(self)            # Initial self.prob parameter as normal
        
        if len(input) == 1:
            GMMDense.Output_Model(self)     # Save Model

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
        SumOps = torch.sum(outputs, dim = 1)
        SumOps = torch.reshape(SumOps, [-1, 1])
        outputs /= SumOps
        #print(outputs)
        return outputs



    def extra_repr(self):
        #Output the io size for visible
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )


        