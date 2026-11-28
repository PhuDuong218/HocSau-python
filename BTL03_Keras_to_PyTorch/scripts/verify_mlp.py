"""Reload both checkpoints and verify measured results against predictions."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mlp_models import GaussianMLP,MNISTMLP
from src.mlp_data import make_mlp_mnist_loaders
from src.train_mlp import epoch_pass,seed_everything

seed_everything(42)
function=torch.load(ROOT/'checkpoints/EXP_01_function.pth',map_location='cpu',weights_only=True)
model=GaussianMLP(function['config']['hidden'])
model.load_state_dict(function['state_dict']); model.eval()
with (ROOT/'results/EXP_01/test_predictions.csv').open() as f:
    saved=list(csv.DictReader(f))
x=torch.tensor([[float(r['x1']),float(r['x2'])] for r in saved],dtype=torch.float32)
y=np.asarray([float(r['target']) for r in saved],dtype=np.float32)
with torch.no_grad(): pred=model(x).numpy().ravel()
np.testing.assert_allclose(pred,[float(r['predicted']) for r in saved],atol=1e-8)
assert np.isclose(np.square(pred-y).mean(),function['metrics']['test_mse'],rtol=1e-5)
mnist=torch.load(ROOT/'checkpoints/T03_mnist.pth',map_location='cpu',weights_only=True)
model=MNISTMLP(); model.load_state_dict(mnist['state_dict']); model.eval()
loaders=make_mlp_mnist_loaders(ROOT/'data/mnist')
loss,accuracy=epoch_pass(model,loaders['test'],nn.CrossEntropyLoss(),torch.device('cpu'),classification=True)
with (ROOT/'results/T03_mnist/predictions.csv').open() as f:
    saved=list(csv.DictReader(f))
pred=[]
with torch.no_grad():
    for x,y in loaders['test']: pred.extend(model(x).argmax(1).tolist())
assert pred==[int(r['predicted']) for r in saved]
assert np.isclose(accuracy,mnist['metrics']['test_accuracy'],atol=1e-12)
assert np.isclose(loss,mnist['metrics']['test_loss'],rtol=1e-5)
cm=np.zeros((10,10),dtype=int)
np.add.at(cm,([int(r['true']) for r in saved],pred),1)
assert cm.sum()==10000 and np.isclose(np.trace(cm)/cm.sum(),accuracy)
result={'status':'passed','function_predictions':100,'mnist_predictions':10000,
        'checkpoint_reload':True,'mnist_test_loss':loss,'mnist_test_accuracy':accuracy}
print(json.dumps(result,indent=2))
