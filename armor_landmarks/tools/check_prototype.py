"""Evaluate a fresh synthetic test seed and render predicted landmarks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from armor.data import SyntheticDataset, read_image, write_image
from armor.inference import load_checkpoint,predict,draw_prediction
from main import evaluate,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='runs/prototype/best.pt')
    parser.add_argument('--config',default='templates/config.json')
    parser.add_argument('--out',default='runs/prototype_check')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(4)
    model,checkpoint=load_checkpoint(args.checkpoint,args.device)
    dataset=SyntheticDataset(args.config,length=280,size=checkpoint['input_size'],seed=987654)
    if dataset.config!=checkpoint['config']:
        raise ValueError('Test template definitions differ from checkpoint')
    metrics=evaluate(model,DataLoader(dataset,batch_size=28),args.device)
    metrics['domain']='synthetic only; not a real-camera benchmark'
    metrics['test_seed']=987654
    metrics['annotations_status']=dataset.config['annotations_status']
    metrics['checkpoint_epoch']=checkpoint['epoch']
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'synthetic_test_metrics.json',metrics)
    cells=[]
    for i in range(6):
        image,target=dataset.sample(i)
        result=predict(model,checkpoint,image,args.device)
        write_json(out/f'warped_{i}.json',result)
        write_image(out/f'warped_{i}_input.png',image)
        overlay=draw_prediction(image,result)
        for j in range(len(dataset.config['classes'][i]['points'])):
            if target['visible'][j]:
                xy=tuple(np.rint(target['points'][j].numpy()).astype(int))
                cv2.drawMarker(overlay,xy,(255,255,0),cv2.MARKER_CROSS,5,1)
        overlay=cv2.resize(overlay,(320,320),interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(overlay,(0,0),(320,30),(25,25,25),-1)
        text=f'GT {i} / Pred {result["class_id"]} ({result["class_score"]:.2f})'
        cv2.putText(overlay,text,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cells.append(overlay)
    sheet=np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])])
    footer=np.full((35,sheet.shape[1],3),25,np.uint8)
    cv2.putText(footer,'Green: valid predictions | Orange: rejected | Cyan crosses: synthetic ground truth',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    write_image(out/'prediction_demo.jpg',np.vstack([sheet,footer]))
    for spec in dataset.config['classes']:
        image=read_image(Path(args.config).parent/spec['image'])
        result=predict(model,checkpoint,image,args.device)
        write_json(out/f'template_{spec["id"]}.json',result)
        write_image(out/f'template_{spec["id"]}.png',draw_prediction(image,result))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
