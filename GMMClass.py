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
        self.Mu.data.uniform_(-1, 1)
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

        sigma_inv = []
        sigma_det = torch.zeros().to(self.device)
        for i in range(0, len(self.Sigma)):
            sigma_inv.append(torch.inverse(self.Sigma[i]).to(self.device))
            print(self.Sigma[i])
            print(torch.det(self.Sigma[i]))
            det = torch.sqrt(torch.max(torch.det(self.Sigma[i]), Zero)).to(self.device)
            print(det)
            if torch.gt(det, epsilon):
                det = torch.Tensor(1).to(self.device)
                sigma_det.append(1/det)
            else:
                sigma_det.append([0])
        print(sigma_det)
        return sigma_inv, torch.Tensor(sigma_det).to(self.device)


    def forward(self, input):
        import math
        Zero        = torch.zeros(1).to(self.device)
                                            # def para =: 0 as tensor
        GMMDense.Unit_prob(self)            # Initial self.prob parameter as normal
        #GMMDense.Output_Model(self)        # Save Model
        sigma_inv, sigma_det = GMMDense.Sigma_Cal(self)
                                            # cal sigma-inverse and sigma-det to calculate
        pi_para     = 0.3989422804014327 * sigma_det
                                            # def para =: 1/sqrt(2 * pi det(sigma))
        pre_para    = torch.mul(pi_para, self.prob)
                                            # p_i \over {2 \pi |\Sigma|}
        pre_para    = torch.reshape(pre_para, [-1, 1])
        exp_val     = -0.5 * torch.exp(torch.mm( torch.mm( (input - mu).t(), sigma_inv), (input - mu))).to(device)
                                            # exp((x-\mu)^t \Sigma^{-1} (x-\mu))
        exp_val     = torch.reshape(exp_val, [-1, 1])

        outputs     = torch.zeros(self.input_features, self.output_features).to(self.device)
        for i in range(0, len(outputs)):
            for j in range(0, len(outputs[i])):
                outputs[i][j] = exp_val[i] * pre_para[j]
        return outputs



    def extra_repr(self):
        #Output the io size for visible
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )


        