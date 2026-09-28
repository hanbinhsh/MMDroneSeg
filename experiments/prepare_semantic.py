"""Audit all original label IDs; recover fine classes on the existing 320/80 split."""
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import openpyxl
from PIL import Image

from .data import ROOT, DATA_ROOT, COLORS, decode_mask
from .train import atomic_json

SOURCE=ROOT.parent/'Drone/semantic_drone_dataset/semantic_drone_dataset'
DEST=ROOT/'.cache/semantic22_960x736'
RUNS=ROOT/'runs/semantic22_960x736'


def main():
    cv2.setNumThreads(0)
    DEST.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    workbook=ROOT.parent/'Drone/colormaps.xlsx'
    rows=list(openpyxl.load_workbook(workbook,data_only=True).active.values)[1:25]
    catalog=[{'id':int(r[4]),'name':r[0],'color':list(r[1:4])} for r in rows]
    assert [r['id'] for r in catalog]==list(range(24))
    ignored=[r['id'] for r in catalog if r['name'] in ('unlabeled','conflicting')]
    assert ignored==[0,23]
    selected=[r for r in catalog if r['id'] not in ignored]
    lookup=np.full(256,255,np.uint8)
    for new,row in enumerate(selected):lookup[row['id']]=new
    group_ast=ast.parse((ROOT.parent/'Drone/classes_dict.txt').read_text())
    grouped=next(ast.literal_eval(n.value) for n in group_ast.body if isinstance(n,ast.Assign) and n.targets[0].id=='grouped_classes')
    coarse_lut=np.zeros(256,np.uint8)
    for key,ids in grouped.items():
        for old in ids:coarse_lut[old]=key
    splits={split:sorted(p.name for p in (DATA_ROOT/folder).glob('*.png')) for split,folder in [('train','original_images'),('val','val_original')]}
    assert len(splits['train'])==320 and len(splits['val'])==80
    assert not set(splits['train']) & set(splits['val'])
    assert set(p.stem for p in (SOURCE/'label_images_semantic').glob('*.png'))=={Path(n).stem for names in splits.values() for n in names}
    raw_counts={s:np.zeros(24,dtype=np.int64) for s in splits}
    resized_counts={s:np.zeros(256,dtype=np.int64) for s in splits}
    records=[];visual_checks=[]
    for split,names in splits.items():
        target=DEST/('label_images_semantic' if split=='train' else 'val_label');target.mkdir(exist_ok=True)
        rgb_root=DATA_ROOT/('original_images' if split=='train' else 'val_original')
        old_masks=DATA_ROOT/('label_images_semantic' if split=='train' else 'val_label')
        for i,name in enumerate(names):
            label_path=SOURCE/'label_images_semantic'/name
            image_path=SOURCE/'original_images'/(Path(name).stem+'.jpg')
            with Image.open(label_path) as im:
                assert im.mode=='L' and im.size==(6000,4000)
                raw=np.asarray(im)
            with Image.open(image_path) as im:assert im.size==(6000,4000)
            assert raw.max()<24
            raw_counts[split]+=np.bincount(raw.reshape(-1),minlength=24)
            resized=cv2.resize(raw,(960,736),interpolation=cv2.INTER_NEAREST)
            mask=lookup[resized]
            resized_counts[split]+=np.bincount(mask.reshape(-1),minlength=256)
            dest=target/name
            Image.fromarray(mask).save(dest)
            with Image.open(old_masks/name) as im:coarse=decode_mask(np.asarray(im.convert('RGB')))
            agreement=float(np.mean(coarse_lut[resized]==coarse))
            assert agreement>.90,(name,'spatial alignment failed',agreement)
            record={'split':split,'name':name,'source_label_sha256':hashlib.sha256(label_path.read_bytes()).hexdigest(),
                    'prepared_mask_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'coarse_label_agreement':agreement}
            records.append(record)
            if i==0 or name=='478.png':
                native=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
                native=cv2.resize(native,(960,736),interpolation=cv2.INTER_AREA)
                small=np.asarray(Image.open(rgb_root/name).convert('RGB'))
                # Original JPEG vs existing resized PNG may use different interpolation.
                mae=float(np.abs(native.astype(float)-small).mean())
                assert mae<20,(name,'RGB correspondence failed',mae)
                visual_checks.append({'name':name,'rgb_resize_mean_absolute_error':mae})
            if (i+1)%40==0:print(f'Audited {split}: {i+1}/{len(names)}',flush=True)
    spec={'name':'semantic22_960x736','classes':[r['name'] for r in selected],
          'colors':[r['color'] for r in selected],'mask_mode':'indexed','ignore_index':255,
          'source_ids':[r['id'] for r in selected],'ignored_source_ids':ignored,'source_catalog':catalog,
          'image_roots':{'train':str(DATA_ROOT/'original_images'),'val':str(DATA_ROOT/'val_original')},
          'protocol':'original_semantic22_resized960x736_crop512_stride384_ignore0_23_no_tta',
          'source_root':str(SOURCE),'label_resize':'OpenCV INTER_NEAREST','resolution_wh':[960,736],
          'split_names':splits,'label_fingerprint':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
          'catalog_sha256':hashlib.sha256(workbook.read_bytes()).hexdigest()}
    path=DEST/'dataset.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==spec,'Existing prepared dataset differs'
    atomic_json(path,spec)
    audit={'spec':spec,'raw_id_pixel_counts':{s:c.tolist() for s,c in raw_counts.items()},
           'prepared_class_pixel_counts':{s:c[:22].tolist() for s,c in resized_counts.items()},
           'ignored_pixels':{s:int(c[255]) for s,c in resized_counts.items()},
           'missing_classes':{s:[selected[i]['name'] for i in range(22) if c[i]==0] for s,c in resized_counts.items()},
           'rgb_correspondence_checks':visual_checks,'records':records}
    atomic_json(RUNS/'dataset_audit.json',audit)
    print('Classes:',spec['classes'])
    print('Missing classes:',audit['missing_classes'])
    print('Minimum coarse-label agreement:',min(r['coarse_label_agreement'] for r in records))
    print('Prepared dataset:',DEST)


if __name__=='__main__':main()
