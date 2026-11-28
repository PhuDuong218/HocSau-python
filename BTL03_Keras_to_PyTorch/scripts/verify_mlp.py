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
from src.mlp_data import gaussian_grid,make_mlp_mnist_loaders
from src.train_mlp import epoch_pass,seed_everything

seed_everything(42)
function=torch.load(ROOT/'checkpoints/T03_function.pth',map_location='cpu',weights_only=True)
model=GaussianMLP(function['config']['hidden'])
model.load_state_dict(function['state_dict']); model.eval()
x,y=gaussian_grid(10)
with torch.no_grad(): pred=model(x).numpy().ravel()
with (ROOT/'results/T03_function/predictions.csv').open() as f:
    saved=list(csv.DictReader(f))
np.testing.assert_allclose(pred,[float(r['predicted']) for r in saved],atol=1e-8)
assert np.isclose(np.square(pred-y.numpy().ravel()).mean(),function['metrics']['test_mse'],rtol=1e-5)
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
cm=np.loadtxt(ROOT/'results/T03_mnist/confusion_matrix.csv',delimiter=',')
assert cm.sum()==10000 and np.isclose(np.trace(cm)/cm.sum(),accuracy)
result={'status':'passed','function_predictions':100,'mnist_predictions':10000,
        'checkpoint_reload':True,'mnist_test_loss':loss,'mnist_test_accuracy':accuracy}
(ROOT/'results/T03_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
