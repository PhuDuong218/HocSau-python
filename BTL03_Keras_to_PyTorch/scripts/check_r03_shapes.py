"""Read-only architecture check of the owner's SegNet; not a training rerun."""
from pathlib import Path
import importlib.util
import json
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reports/R03_figures';out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('segnet_owner',ROOT/'references/R03/segnet_owner_class.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2)
model=module.HandSegNet().eval()
shapes={};handles=[]
for name in ['enc1','enc2','enc3','dec3','dec2','dec1']:
 def hook(_module,_inputs,output,name=name): shapes[name]=list(output.shape)
 handles.append(getattr(model,name).register_forward_hook(hook))
with torch.no_grad():
 output=model(torch.zeros(2,1,96,96))
 loss=nn.CrossEntropyLoss()(output,torch.zeros(2,96,96,dtype=torch.long))
for handle in handles:handle.remove()
expected={'enc1':[2,64,48,48],'enc2':[2,128,24,24],'enc3':[2,256,12,12],
          'dec3':[2,128,24,24],'dec2':[2,64,48,48],'dec1':[2,7,96,96]}
assert shapes==expected and torch.isfinite(loss)
params=sum(p.numel() for p in model.parameters())
assert params==2291207
result={'status':'passed','check':'synthetic forward only; no owner dataset or training rerun',
        'input':[2,1,96,96],'intermediate_shapes':shapes,'params':params,
        'output_logits':list(output.shape),'target_dtype':'torch.long','target_shape':[2,96,96]}
(ROOT/'reports/R03_shape_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def diagram(name,items,columns):
 rows=(len(items)+columns-1)//columns
 fig,ax=plt.subplots(figsize=(12,rows*1.5))
 ax.set(xlim=(0,columns*3),ylim=(0,rows*1.5));ax.axis('off')
 positions=[]
 for i,(title,shape) in enumerate(items):
  row=i//columns; col=i%columns
  if row%2:col=columns-1-col
  x=col*3+.15;y=(rows-1-row)*1.5+.3
  box=FancyBboxPatch((x,y),2.7,.9,boxstyle='round,pad=0.03',facecolor='#edf3f7',edgecolor='#334455')
  ax.add_patch(box);ax.text(x+1.35,y+.61,title,ha='center',va='center',fontsize=10,weight='bold')
  ax.text(x+1.35,y+.27,shape,ha='center',va='center',fontsize=10)
  positions.append((x,y))
 for i in range(len(positions)-1):
  x,y=positions[i];xx,yy=positions[i+1]
  if y==yy:
   start=(x+2.74,y+.45) if xx>x else (x-.04,y+.45)
   end=(xx-.03,yy+.45) if xx>x else (xx+2.73,yy+.45)
  else:start=(x+1.35,y-.03);end=(xx+1.35,yy+.93)
  ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#334455','lw':1.5})
 fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=180);fig.savefig(out/f'{name}.svg');plt.close(fig)
 svg=out/f'{name}.svg'
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')

diagram('segnet_tensor_flow',[
 ('Depth input','[B,1,96,96]'),('Encoder 1','[B,64,48,48]'),('Encoder 2','[B,128,24,24]'),('Encoder 3','[B,256,12,12]'),
 ('Decoder 3','[B,128,24,24]'),('Decoder 2','[B,64,48,48]'),('Decoder 1 logits','[B,7,96,96]'),('argmax over classes','[B,96,96]')],4)
diagram('lstm_tensor_flow',[
 ('7-frame input','[B,7,1,28,28]'),('Fold batch and time','[B*7,1,28,28]'),('Shared CNN','[B*7,256,3,3]'),
 ('Flatten and unfold','[B,7,2304]'),('LSTM 1 + LSTM 2','[B,7,128] -> [B,128]'),('Dropout .22 + Linear','[B,1] predicted sum')],3)
print(json.dumps(result,indent=2))
