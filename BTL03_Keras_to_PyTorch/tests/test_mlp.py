"""T03 correctness checks without network or large datasets."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.mlp_models import GaussianMLP, MNISTMLP, parameter_count
from src.mlp_data import gaussian_grid, make_mlp_mnist_loaders
from src.train_mlp import epoch_pass, seed_everything


class MLPTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)

    def test_dense_conversion_matches_numpy_keras_orientation(self):
        # Keras Dense kernel [in,out] must be transposed for nn.Linear [out,in].
        rng=np.random.default_rng(5)
        w1=rng.normal(size=(2,4)).astype('float32')
        b1=rng.normal(size=4).astype('float32')
        w2=rng.normal(size=(4,1)).astype('float32')
        b2=rng.normal(size=1).astype('float32')
        x=rng.normal(size=(7,2)).astype('float32')
        sigmoid=lambda z: 1/(1+np.exp(-z))
        expected=sigmoid(sigmoid(x@w1+b1)@w2+b2)
        model=GaussianMLP(4)
        with torch.no_grad():
            model.network[0].weight.copy_(torch.from_numpy(w1.T))
            model.network[0].bias.copy_(torch.from_numpy(b1))
            model.network[2].weight.copy_(torch.from_numpy(w2.T))
            model.network[2].bias.copy_(torch.from_numpy(b2))
        np.testing.assert_allclose(model(torch.from_numpy(x)).detach(),expected,atol=1e-7)

    def test_mnist_shape_params_logits_and_loss(self):
        model=MNISTMLP().eval()
        x=torch.randn(5,1,28,28)
        logits=model(x)
        self.assertEqual(tuple(logits.shape),(5,10))
        self.assertEqual(parameter_count(model),669706)
        self.assertFalse(any(isinstance(m,nn.Softmax) for m in model.modules()))
        labels=torch.tensor([0,1,2,3,4],dtype=torch.long)
        manual=(-logits.log_softmax(1)[torch.arange(5),labels]).mean()
        torch.testing.assert_close(nn.CrossEntropyLoss()(logits,labels),manual)
        torch.testing.assert_close(model(x),model(x))

    def test_gaussian_grid_and_target_shape(self):
        x,y=gaussian_grid(32)
        self.assertEqual(tuple(x.shape),(1024,2))
        self.assertEqual(tuple(y.shape),(1024,1))
        expected=np.exp(-9)/(2*np.pi)
        self.assertAlmostEqual(y[0].item(),expected,places=9)
        self.assertEqual(parameter_count(GaussianMLP(128)),513)

    def test_training_step_and_reload(self):
        model=GaussianMLP(8)
        x=torch.randn(10,2); y=torch.zeros(10,1)
        loss=nn.MSELoss(); opt=torch.optim.SGD(model.parameters(),lr=0.1)
        before=loss(model(x),y).item()
        opt.zero_grad(); loss(model(x),y).backward(); opt.step()
        self.assertLess(loss(model(x),y).item(),before)
        stream=io.BytesIO(); torch.save(model.state_dict(),stream); stream.seek(0)
        reloaded=GaussianMLP(8); reloaded.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(model(x),reloaded(x))

    def test_epoch_loss_weighted_by_samples(self):
        model=GaussianMLP(8).eval()
        x,y=gaussian_grid(3)
        loader=DataLoader(TensorDataset(x,y),batch_size=4)
        value,_=epoch_pass(model,loader,nn.MSELoss(),torch.device('cpu'))
        self.assertAlmostEqual(value,nn.MSELoss()(model(x),y).item(),places=7)

    def test_shared_dataset_adapter_preserves_raw_scaling(self):
        data=torch.arange(3*28*28,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3,28,28)
        targets=torch.tensor([1,2,3])
        source=SimpleNamespace(data=data.clone(),targets=targets.clone())
        shared={'train':SimpleNamespace(dataset=source),'test':SimpleNamespace(dataset=source)}
        with patch('src.mlp_data.make_mnist_loaders',return_value=shared):
            loaders=make_mlp_mnist_loaders('unused',batch_size=2)
        x,y=next(iter(loaders['test']))
        self.assertEqual(tuple(x.shape),(2,1,28,28))
        self.assertEqual(y.dtype,torch.long)
        torch.testing.assert_close(x[:,0],data[:2].float()/255)
        torch.testing.assert_close(source.data,data)

if __name__=='__main__': unittest.main()
