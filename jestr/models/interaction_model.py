
import torch
import torch.nn as nn

class INTER_MLP2(nn.Module):
    def __init__(self, params):
        super(INTER_MLP2, self).__init__()
        self.dropout = nn.Dropout(params['fc_dropout'])
        self.fp_fc1 = nn.Linear(params['final_embedding_dim'] * 2, params['final_embedding_dim'])
        self.fp_fc2 = nn.Linear(params['final_embedding_dim'], params['final_embedding_dim'] // 2)
        self.fp_fc3 = nn.Linear(params['final_embedding_dim'] // 2, params['final_embedding_dim'] // 4)
        self.fp_fc4 = nn.Linear(params['final_embedding_dim'] // 4, params['final_embedding_dim'] // 8)
        self.fp_fc5 = nn.Linear(params['final_embedding_dim'] // 8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, mol, spec):
        
        y_cat = torch.cat((mol, spec), 1) 
            
        h = self.fp_fc1(y_cat)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc3(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fp_fc4(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.fp_fc5(h)
        z_interaction = self.sigmoid(h)
       
        return z_interaction