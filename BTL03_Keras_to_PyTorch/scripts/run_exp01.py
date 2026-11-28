"""EXP_01 from the tracking sheet: random [-1,1] input, SGD .01, B32, E50."""
from pathlib import Path
import sys
import time
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.mlp_models import GaussianMLP,parameter_count
from src.train_mlp import seed_everything,epoch_pass,write_csv,write_json,save_checkpoint,provenance
import matplotlib.pyplot as plt

seed_everything(42)
rng=np.random.default_rng(42)
def make_data(n):
    x=rng.uniform(-1,1,(n,2)).astype('float32')
    y=(np.exp(-np.square(x).sum(1)/2)/(2*np.pi)).astype('float32')[:,None]
    return torch.from_numpy(x),torch.from_numpy(y)
x,y=make_data(1024);xt,yt=make_data(100)
folder=ROOT/'results/EXP_01';folder.mkdir(parents=True,exist_ok=True)
loader=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42))
test=DataLoader(TensorDataset(xt,yt),batch_size=100)
model=GaussianMLP(16);optimizer=torch.optim.SGD(model.parameters(),lr=.01)
loss=nn.MSELoss();history=[];start=time.perf_counter()
for epoch in range(1,51):
    tr,_=epoch_pass(model,loader,loss,torch.device('cpu'),optimizer)
    te,_=epoch_pass(model,test,loss,torch.device('cpu'))
    history.append({'epoch':epoch,'train_mse':tr,'test_mse':te})
elapsed=time.perf_counter()-start
model.eval()
with torch.no_grad():
    pred=model(xt)
    train_final=float(loss(model(x),y))
    test_final=float(loss(pred,yt))
metrics={'run_id':'EXP_01','train_mse':train_final,'test_mse':test_final,
         'test_mae':float((pred-yt).abs().mean()),'params':parameter_count(model),
         'output_shape':'[B,1]','training_seconds':elapsed}
config={'seed':42,'input_range':[-1,1],'train_samples':1024,'test_samples':100,
        'target':'exp(-(x1^2+x2^2)/2)/(2*pi), Gaussian from original notebook',
        'sampling':'independent random uniform x1,x2; no grid',
        'hidden':16,'batch_size':32,'epochs':50,'optimizer':'SGD','lr':.01,'loss':'MSELoss',
        'device':'cpu','note':'Separate run matching tracking-sheet hyperparameters; original grid sweeps retained in T03_function.'}
save_checkpoint(ROOT/'checkpoints/EXP_01_function.pth',model,config,metrics)
state=torch.load(ROOT/'checkpoints/EXP_01_function.pth',weights_only=True)
reloaded=GaussianMLP(16);reloaded.load_state_dict(state['state_dict']);reloaded.eval()
with torch.no_grad():torch.testing.assert_close(reloaded(xt),pred)
write_json(folder/'metrics.json',{**metrics,'config':config,'checkpoint_reload':'passed'})
write_json(folder/'metadata.json',provenance())
write_csv(folder/'history.csv',history)
write_csv(folder/'train_data.csv',[{'x1':float(a),'x2':float(b),'target':float(v)} for (a,b),v in zip(x.tolist(),y[:,0].tolist())])
write_csv(folder/'test_predictions.csv',[{'x1':float(a),'x2':float(b),'target':float(v),'predicted':float(p)} for (a,b),v,p in zip(xt.tolist(),yt[:,0].tolist(),pred[:,0].tolist())])
(folder/'train.log').write_text('\n'.join(f"epoch={r['epoch']} train_mse={r['train_mse']:.9f} test_mse={r['test_mse']:.9f}" for r in history)+'\n'+json.dumps(metrics),encoding='utf-8')
fig,ax=plt.subplots(figsize=(7,3.5))
for key in ['train_mse','test_mse']:ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
ax.set(xlabel='Epoch',ylabel='MSE',title='EXP_01: SGD .01, batch 32, 50 epochs');ax.legend()
fig.tight_layout();fig.savefig(folder/'loss_curve.png',dpi=150);plt.close(fig)
print(json.dumps(metrics,indent=2),flush=True)
