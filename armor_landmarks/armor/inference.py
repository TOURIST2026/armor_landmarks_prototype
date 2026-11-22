from pathlib import Path
import cv2
import numpy as np
import torch

from .data import image_tensor
from .geometry import letterbox, heatmap_to_network, transform_points, center_offsets
from .model import ArmorNet


def decode_heatmaps(logits, radius=2):
    """Numpy local soft-argmax. Returns network-pixel xy and uncalibrated mass."""
    points, scores = [], []
    for heatmap in logits:
        h,w = heatmap.shape
        y,x = np.unravel_index(np.argmax(heatmap),heatmap.shape)
        x0,x1,y0,y1 = max(0,x-radius),min(w,x+radius+1),max(0,y-radius),min(h,y+radius+1)
        values = np.exp(heatmap-heatmap.max())
        local = values[y0:y1,x0:x1]
        yy,xx = np.mgrid[y0:y1,x0:x1]
        points.append([(local*xx).sum()/local.sum(),(local*yy).sum()/local.sum()])
        scores.append(float(local.sum()/values.sum()))
    return heatmap_to_network(np.array(points,np.float32)), np.array(scores)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    model = ArmorNet(len(checkpoint['config']['classes']),checkpoint['config']['max_points'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


def predict(model, checkpoint, image, device='cpu', class_threshold=0.7, point_threshold=0.3, roi_to_frame=None):
    config = checkpoint['config']
    size = checkpoint['input_size']
    net_image, matrix = letterbox(image,size)
    with torch.inference_mode():
        cls,heat,vis = model(image_tensor(net_image)[None].to(device))
    probs = cls.softmax(-1)[0].cpu().numpy()
    cid = int(probs.argmax())
    h,w = image.shape[:2]
    result = {'class_id':cid,'class_name':'invalid' if cid == len(config['classes']) else config['classes'][cid]['name'],
              'class_score':float(probs[cid]), 'class_probabilities':probs.tolist(),
              'roi_width':w,'roi_height':h,'coordinate_convention':'pixels; x right, y down; center=((W-1)/2,(H-1)/2)',
              'annotations_status':config.get('annotations_status','unknown'),
              'checkpoint_status':checkpoint.get('status','unvalidated'),
              'thresholds_calibrated':False,'accepted':False,'points':[]}
    if cid == len(config['classes']) or probs[cid] < class_threshold:
        result['reason'] = 'invalid_or_uncertain_class'
        return result
    spec = config['classes'][cid]['points']
    points, mass = decode_heatmaps(heat[0,cid,:len(spec)].cpu().numpy())
    points = transform_points(points,np.linalg.inv(matrix))
    offsets = center_offsets(points,w,h)
    visibility = vis[0,cid,:len(spec)].sigmoid().cpu().numpy()
    for i,item in enumerate(spec):
        score = float(visibility[i]*mass[i])
        inside = bool(0 <= points[i,0] <= w-1 and 0 <= points[i,1] <= h-1)
        point = {'point_id':item['id'],'uv_roi':points[i].tolist(),'offset_from_roi_center':offsets[i].tolist(),
                 'visibility_score':float(visibility[i]),'local_probability_mass':float(mass[i]),
                 'point_score':score,'valid':inside and score >= point_threshold}
        if roi_to_frame is not None:
            point['uv_frame'] = transform_points([points[i]],roi_to_frame)[0].tolist()
        result['points'].append(point)
    valid = np.array([p['uv_roi'] for p in result['points'] if p['valid']],np.float32).reshape(-1,2)
    result['accepted'] = bool(len(valid)>=4 and cv2.contourArea(cv2.convexHull(valid)) > 1.0)
    result['reason'] = 'ok' if result['accepted'] else 'insufficient_reliable_spread_points'
    return result


def draw_prediction(image, result):
    canvas = image.copy()
    for point in result['points']:
        xy = tuple(np.rint(point['uv_roi']).astype(int))
        color = (0,220,0) if point['valid'] else (0,140,255)
        cv2.circle(canvas,xy,2,color,-1)
        cv2.putText(canvas,str(point['point_id']),xy,cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)
    return canvas
