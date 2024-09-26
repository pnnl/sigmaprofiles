import torch
import torch.nn as nn

checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))

regression_head = nn.Linear(512, 1)
state_dict = checkpoint['model_state_dict']

state_dict.pop('classification_head.dense.weight')
state_dict.pop('classification_head.dense.bias')
state_dict.pop('classification_head.out_proj.weight')
state_dict.pop('classification_head.out_proj.bias')

state_dict['regression_head.weight'] = regression_head.weight
state_dict['regression_head.bias'] = regression_head.bias

torch.save({'model': state_dict}, 'modified_model.pth')