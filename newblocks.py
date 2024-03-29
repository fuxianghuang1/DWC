import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class Inception1(nn.Module):   
  def __init__(self, in_dim, out_dim): 
    super(Inception1, self).__init__()
    self.branch0 = nn.Sequential(nn.Linear(in_dim, (out_dim//8) * 3), nn.ReLU(), nn.Dropout(0.5))
    
    self.branch1 = nn.Sequential(nn.Linear(in_dim, out_dim//2), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(out_dim//2, out_dim//4), nn.ReLU(), nn.Dropout(0.5)  )
                                 
    self.branch2 = nn.Sequential(nn.Linear(in_dim, out_dim//2), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(out_dim//2, out_dim//4), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(out_dim//4, out_dim//4), nn.ReLU(), nn.Dropout(0.5))
                                 
    self.branch3 = nn.Sequential(nn.Linear(in_dim, out_dim//2), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(out_dim//2, out_dim//8), nn.ReLU(), nn.Dropout(0.5) )                                   
  def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), -1)
        return out 
        
class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = self.relu(x)
        return x

class Inception2(nn.Module):

    def __init__(self, in_dim=2048, out_dim=1024):
        super(Inception2, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_dim, (out_dim//8) * 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_dim, out_dim//2, kernel_size=1, stride=1),
            nn.ReLU(),
            BasicConv2d(out_dim//2, out_dim//4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_dim, out_dim//4, kernel_size=1, stride=1),
            nn.ReLU(),
            BasicConv2d(out_dim//4, out_dim//8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            BasicConv2d(out_dim//8, out_dim//4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_dim, out_dim//8, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Inception3(nn.Module):

    def __init__(self, in_dim=2048, out_dim=1024):
        super(Inception3, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_dim, (out_dim//8) * 3, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_dim, out_dim//2, kernel_size=1, stride=1),
            nn.ReLU(),
            BasicConv2d(out_dim//2, out_dim//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_dim, out_dim//4, kernel_size=1, stride=1),
            nn.ReLU(),
            BasicConv2d(out_dim//4, out_dim//8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            BasicConv2d(out_dim//8, out_dim//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_dim, out_dim//8, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
        
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        n_dim = len(x.size())
        if n_dim == 3:
            if self.p == 1:
                return x.mean(dim=[-1])
            elif self.p == float('inf'):
                return torch.flatten(F.adaptive_max_pool1d(x, 1), start_dim=1)
            else:
                return torch.flatten(F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(1./self.p), start_dim=1)
        elif n_dim == 4:
            if self.p == 1:
                return x.mean(dim=[-1, -2])
            elif self.p == float('inf'):
                return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
            #return LF.gem(x, p=self.p, eps=self.eps)
            else:
                return torch.flatten(F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p), start_dim=1)
                
                
class NormalizationLayer(torch.nn.Module):
  """Class for normalization layer."""
  def __init__(self, normalize_scale=1.0, learn_scale=True):
    super(NormalizationLayer, self).__init__()
    self.norm_s = float(normalize_scale)
    if learn_scale:
      self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

  def forward(self, x):
    features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
    return features
    
class GlobalAvgPool2d(torch.nn.Module):

  def forward(self, x):
    return F.adaptive_avg_pool2d(x, (1, 1))    