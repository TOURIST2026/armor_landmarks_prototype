"""Run from this directory: python main.py --help."""
import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from armor.data import SyntheticDataset, RealDataset, load_templates, read_image, write_image
from armor.model import ArmorNet, losses
from armor.inference import load_checkpoint, predict, draw_prediction, decode_heatmaps


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def move_target(target, device):
    return {key:value.to(device) for key,value in target.items()}


def evaluate(model, loader, device):
    model.eval()
    count,correct,total_loss = 0,0,0.
    errors, joint_hits = [], []
    per_class = {}
    enough, eligible = 0,0
    with torch.inference_mode():
        for images,target in loader:
            outputs = model(images.to(device))
            loss,_ = losses(outputs,move_target(target,device))
            total_loss += float(loss)*len(images)
            predicted = outputs[0].argmax(-1).cpu().numpy()
            labels = target['class_id'].numpy()
            correct += int((predicted==labels).sum())
            count += len(images)
            for i,cid in enumerate(labels):
                stats = per_class.setdefault(str(cid),{'samples':0,'correct':0,'visible_points':0,'joint_pck3_hits':0})
                stats['samples'] += 1
                stats['correct'] += int(predicted[i]==cid)
                if cid == outputs[1].shape[1]:
                    continue
                mask = target['visible'][i].numpy().astype(bool)
                if not mask.any():
                    continue
                xy,_ = decode_heatmaps(outputs[1][i,cid].cpu().numpy())
                error = np.linalg.norm((xy[mask]-target['points'][i].numpy()[mask])/target['roi_scale'][i].numpy(),axis=1)
                hits = (error<=3)&(predicted[i]==cid)
                errors.extend(error.tolist())
                joint_hits.extend(hits.tolist())
                stats['visible_points'] += len(error)
                stats['joint_pck3_hits'] += int(hits.sum())
                if len(error)>=4:
                    eligible += 1
                    enough += int(hits.sum()>=4)
    return {'samples':count,'loss':total_loss/max(count,1),'classification_accuracy':correct/max(count,1),
            'coordinate_space':'original ROI pixels; synthetic ROI size equals network input size',
            'oracle_class_point_error_mean':float(np.mean(errors)) if errors else None,
            'oracle_class_point_error_p95':float(np.percentile(errors,95)) if errors else None,
            'joint_class_pck3':float(np.mean(joint_hits)) if joint_hits else None,
            'at_least_four_correct_points_rate':enough/eligible if eligible else None,
            'note':'oracle metrics select ground-truth class head; joint metrics count class errors as failures; visibility filtering not applied',
            'per_class':per_class}


def train(args):
    config = load_templates(args.config)
    if config.get('annotations_status') != 'reviewed' and not args.allow_draft:
        raise ValueError('Draft annotations: review in annotate.html or explicitly pass --allow-draft for a prototype run')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device
    synthetic = SyntheticDataset(args.config,args.samples,args.size,args.seed)
    data = synthetic
    if args.real_train:
        data = ConcatDataset([synthetic,RealDataset(args.real_train,config,args.size)])
    val = RealDataset(args.real_val,config,args.size) if args.real_val else SyntheticDataset(args.config,args.val_samples,args.size,args.seed+100000)
    loader = DataLoader(data,batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader = DataLoader(val,batch_size=args.batch,num_workers=0)
    if args.init:
        model,previous = load_checkpoint(args.init,device)
        if previous['config'] != config or previous['input_size'] != args.size:
            raise ValueError('Initialization checkpoint config/size mismatch; do not silently reuse changed point IDs')
        model.train()
    else:
        model = ArmorNet(len(config['classes']),config['max_points'],args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    best,step = float('inf'),0
    started = time.perf_counter()
    log = []
    for epoch in range(args.epochs):
        synthetic.epoch = epoch
        model.train()
        train_loss,seen = 0.,0
        for images,target in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(device))
            loss,parts = losses(outputs,move_target(target,device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            optimizer.step()
            step += 1
            train_loss += float(loss.detach())*len(images)
            seen += len(images)
            if step%10 == 0:
                print(json.dumps({'step':step,'loss':float(loss.detach()),'parts':{k:float(v) for k,v in parts.items()}}),flush=True)
            if args.max_steps and step>=args.max_steps:
                break
        metrics = evaluate(model,val_loader,device)
        record = {'epoch':epoch+1,'step':step,'train_loss':train_loss/max(seen,1),'validation':metrics}
        log.append(record)
        status = 'smoke_test_only' if args.max_steps else ('real_validation_required' if not args.real_val else 'threshold_calibration_required')
        checkpoint = {'model':model.state_dict(),'config':config,'input_size':args.size,'epoch':epoch+1,'step':step,
                      'status':status,'validation_domain':'real' if args.real_val else 'synthetic','metrics':metrics}
        torch.save(checkpoint,out/'last.pt')
        if metrics['loss']<best:
            best = metrics['loss']
            torch.save(checkpoint,out/'best.pt')
        write_json(out/'history.json',{'seconds':time.perf_counter()-started,'arguments':vars(args),'epochs':log})
        print(json.dumps(record),flush=True)
        schedule.step()
        if args.max_steps and step>=args.max_steps:
            break


def preview(args):
    data = SyntheticDataset(args.config,args.count,args.size,args.seed)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    cells,records = [],[]
    for i in range(args.count):
        image,target = data.sample(i)
        filename = f'sample_{i:03d}.png'
        write_image(out/filename,image)
        cid = int(target['class_id'])
        spec = data.config['classes'][cid]['points'] if cid<len(data.config['classes']) else []
        records.append({'image':filename,'class_id':cid,'points':[{'id':p['id'],'xy':target['points'][j].tolist(),
                        'visible':bool(target['visible'][j])} for j,p in enumerate(spec)]})
        overlay = image.copy()
        for j,p in enumerate(spec):
            if target['visible'][j]:
                xy = tuple(np.rint(target['points'][j].numpy()).astype(int))
                cv2.circle(overlay,xy,2,(0,255,0),-1)
                cv2.putText(overlay,str(p['id']),xy,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
        cv2.putText(overlay,f'class {cid}',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
        cells.append(overlay)
    while len(cells)%7:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[i:i+7]) for i in range(0,len(cells),7)])
    write_image(out/'contact_sheet.jpg',sheet)
    (out/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    print(f'Generated {args.count} samples, labels and contact_sheet.jpg in {out}')


def inference(args):
    torch.set_num_threads(args.threads)
    model,checkpoint = load_checkpoint(args.checkpoint,args.device)
    image = read_image(args.image)
    frame_matrix = None
    if args.roi_to_frame:
        frame_matrix = np.asarray(json.loads(Path(args.roi_to_frame).read_text()),np.float64).reshape(3,3)
    result = predict(model,checkpoint,image,args.device,args.class_threshold,args.point_threshold,frame_matrix)
    write_json(args.out,result)
    write_image(Path(args.out).with_suffix('.png'),draw_prediction(image,result))
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('preview')
    p.add_argument('--config',default='templates/config.json')
    p.add_argument('--count',type=int,default=28)
    p.add_argument('--size',type=int,default=192)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--out',default='runs/preview')
    p = sub.add_parser('train')
    p.add_argument('--config',default='templates/config.json')
    p.add_argument('--out',default='runs/train')
    p.add_argument('--size',type=int,default=192)
    p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--val-samples',type=int,default=280)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--pretrained',action='store_true')
    p.add_argument('--allow-draft',action='store_true')
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--init')
    p.add_argument('--real-train')
    p.add_argument('--real-val')
    p = sub.add_parser('predict')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--out',default='runs/prediction.json')
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--class-threshold',type=float,default=0.7)
    p.add_argument('--point-threshold',type=float,default=0.3)
    p.add_argument('--roi-to-frame',help='JSON file containing 3x3 transform from raw ROI to original frame')
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--out',default='runs/evaluation.json')
    p = sub.add_parser('export')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',default='runs/model.ts')
    args = parser.parse_args()
    if hasattr(args,'size') and (args.size<64 or args.size%32):
        parser.error('--size must be a multiple of 32 and >=64')
    if args.command=='train':
        train(args)
    elif args.command=='preview':
        preview(args)
    elif args.command=='predict':
        inference(args)
    elif args.command=='evaluate':
        torch.set_num_threads(4)
        model,checkpoint = load_checkpoint(args.checkpoint,args.device)
        data = RealDataset(args.manifest,checkpoint['config'],checkpoint['input_size'])
        report = evaluate(model,DataLoader(data,batch_size=args.batch),args.device)
        write_json(args.out,report)
        print(json.dumps(report,indent=2))
    elif args.command=='export':
        torch.set_num_threads(4)
        model,checkpoint = load_checkpoint(args.checkpoint)
        sample = torch.zeros(1,3,checkpoint['input_size'],checkpoint['input_size'])
        traced = torch.jit.trace(model,sample)
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        traced.save(args.out)
        for a,b in zip(model(sample),traced(sample)):
            torch.testing.assert_close(a,b)
        write_json(str(args.out)+'.json',{'config':checkpoint['config'],'input_size':checkpoint['input_size'],
                   'outputs':['class_logits','heatmap_logits','visibility_logits'],'preprocessing':'see armor/geometry.py and armor/data.py',
                   'checkpoint_status':checkpoint['status']})
        print(f'Exported and checked TorchScript: {args.out}')


if __name__ == '__main__':
    main()
