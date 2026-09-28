import pytest
import torch
from experiments.inference import sliding_logits
from experiments.models import Objective


def test_sliding_arbitrary_class_count():
    class Model(torch.nn.Module):
        def forward(self,batch):
            return {'logits':torch.cat([batch['image'][:,:1]+i for i in range(22)],dim=1)}
    x={'image':torch.randn(1,3,41,63)}
    assert torch.allclose(sliding_logits(Model(),x,32,24,False),Model()(x)['logits'],atol=1e-5)


@pytest.mark.parametrize('name',['ours_no_text','afenet','logcan','d2ls'])
def test_void_crop_has_finite_zero_loss(name):
    logits=torch.randn(2,22,8,8,requires_grad=True)
    target=torch.full((2,8,8),255,dtype=torch.long)
    loss=Objective(name)({'logits':logits},target)
    assert loss.item()==0
    loss.backward();assert logits.grad.abs().sum().item()==0
