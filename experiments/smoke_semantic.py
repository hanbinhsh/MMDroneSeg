"""One real 512-pixel training step per model, no checkpoints written."""
import gc
import json
import time
import torch
from .data import DroneData
from .models import build_model, batch_needs, Objective
from .train import seed_all, make_optimizer, atomic_json
from .prepare_semantic import DEST, RUNS
from .inference import sliding_logits


def main(models=None, report_path=None, crop_kwargs=None):
    torch.set_num_threads(4)
    results=[]
    for name in (models or ['ours_no_text','afenet','logcan','d2ls']):
        seed_all(42);torch.cuda.reset_peak_memory_stats()
        train=DroneData(DEST,'train',**batch_needs(name), **((crop_kwargs or {}) if name == 'ours_no_text' else {}))
        samples=[train[i] for i in range(2)]
        batch={key:torch.stack([s[key] for s in samples]).cuda() for key in samples[0] if key!='name'}
        model=build_model(name,True,22).cuda().train()
        optimizer,_=make_optimizer(model,name,150)
        criterion=Objective(name).cuda();scaler=torch.cuda.amp.GradScaler()
        started=time.perf_counter()
        # Initial AMP scale may overflow; exercise its normal backoff until an
        # actual finite-gradient optimizer step occurs, not merely a skipped step.
        for attempt in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                outputs=model(batch)
                assert outputs['logits'].shape==(2,22,512,512)
                loss=criterion(outputs,batch['mask'])
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if finite:torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            if finite:break
        else:raise AssertionError(f'{name}: gradients remain non-finite after AMP backoff')
        model.eval()
        val=DroneData(DEST,'val',**batch_needs(name))[0]
        vb={key:value.unsqueeze(0).cuda() for key,value in val.items() if key not in ('mask','name')}
        pred=sliding_logits(model,vb)
        assert pred.shape==(1,22,736,960) and torch.isfinite(pred).all()
        result={'model':name,'train_loss':loss.item(),'amp_backoff_attempts':attempt,'amp_scale':scaler.get_scale(),'seconds':time.perf_counter()-started,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20,'output_shape':list(pred.shape)}
        results.append(result);print(json.dumps(result),flush=True)
        atomic_json(report_path or RUNS/'smoke_results.json',results)
        del model,optimizer,criterion,scaler,outputs,loss,batch,pred,vb,samples
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
