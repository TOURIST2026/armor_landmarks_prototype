"""PyCharm: select the existing yolo interpreter, then right-click Run demo."""
import json
from pathlib import Path

import torch
from armor.data import read_image,write_image
from armor.inference import load_checkpoint,predict,draw_prediction

ROOT=Path(__file__).resolve().parent
# Change this to your cropped armor image. It may be an absolute path.
IMAGE=ROOT/'runs/augmentation/sample_018.png'
CHECKPOINT=ROOT/'runs/my_train/best.pt'


if __name__=='__main__':
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,checkpoint=load_checkpoint(CHECKPOINT,device)
    image=read_image(IMAGE)
    result=predict(model,checkpoint,image,device)
    output=ROOT/'runs/demo'
    output.mkdir(parents=True,exist_ok=True)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_image(output/'result.png',draw_prediction(image,result))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(f'Output: {output}')
    print('Prototype uses draft landmark IDs and synthetic-only training; review IDs before PnP integration.')
