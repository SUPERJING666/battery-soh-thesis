import torch
from torch import nn

def backbone():
    return nn.Sequential(nn.Conv1d(1,16,5,5),nn.ReLU(),nn.MaxPool1d(2,2),nn.Conv1d(16,32,5,2),nn.ReLU(),nn.Conv1d(32,32,3),nn.ReLU(),nn.Conv1d(32,64,3),nn.ReLU(),nn.MaxPool1d(2,1))

class Net(nn.Module):
    def __init__(self,kind,curve_dim=151,hi_dim=3,input_layout='legacy'):
        super().__init__();self.kind=kind;self.input_layout=input_layout
        self.cnn_conditions=2 if kind=='CNN' and curve_dim==153 and input_layout=='fair_v3' else 0
        if kind in ['MLP','conditions']:
            self.hhead=nn.Sequential(nn.Linear(hi_dim,32),nn.ReLU(),nn.Linear(32,1),nn.ReLU());return
        self.conv=backbone()
        with torch.no_grad():flat=self.conv(torch.zeros(1,1,curve_dim-self.cnn_conditions+(hi_dim if kind=='Early' else 0))).numel()
        flat+=self.cnn_conditions
        if kind in ['CNN','Early']:
            self.head=nn.Sequential(nn.Linear(flat,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))
        elif kind=='Middle':
            self.chead=nn.Sequential(nn.Linear(flat,32),nn.ReLU());self.hhead=nn.Sequential(nn.Linear(hi_dim,32),nn.ReLU())
            self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))
        elif kind=='Late':
            self.chead=nn.Sequential(nn.Linear(flat,32),nn.ReLU(),nn.Linear(32,1),nn.ReLU());self.hhead=nn.Sequential(nn.Linear(hi_dim,32),nn.ReLU(),nn.Linear(32,1),nn.ReLU())
            self.head=nn.Sequential(nn.Linear(2,8),nn.ReLU(),nn.Linear(8,1))
        else:raise ValueError(kind)

    def initialize(self,mean_y):
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
        linear=[m for m in self.modules() if isinstance(m,nn.Linear)]
        nn.init.constant_(linear[-1].bias,float(mean_y))

    def forward(self,c,h):
        if self.kind in ['MLP','conditions']:return self.hhead(h).flatten()
        condition=None
        if self.cnn_conditions:condition=c[:,-2:];c=c[:,:-2]
        if self.kind=='Early':c=torch.cat([h,c],dim=1) if self.input_layout=='fair_v3' else torch.cat([c,h],dim=1)
        f=self.conv(c.unsqueeze(1)).flatten(1)
        if condition is not None:f=torch.cat([f,condition],dim=1)
        if self.kind in ['Middle','Late']:f=torch.cat([self.chead(f),self.hhead(h)],dim=1)
        y=self.head(f)
        if self.kind!='CNN':y=torch.relu(y)
        return y.flatten()
