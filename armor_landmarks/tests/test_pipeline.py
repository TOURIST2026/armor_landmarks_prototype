import json
from pathlib import Path
import numpy as np
import torch

from armor.geometry import letterbox, transform_points, center_offsets, heatmap_to_network, network_to_heatmap
from armor.data import SyntheticDataset, RealDataset, write_image
from armor.model import ArmorNet, losses
from armor.inference import decode_heatmaps, load_checkpoint, predict

CONFIG = Path(__file__).resolve().parents[1]/'templates/config.json'
torch.set_num_threads(2)


def test_coordinate_roundtrip_non_square():
    image = np.zeros((73,141,3),np.uint8)
    _,matrix = letterbox(image)
    points = np.array([[0,0],[140,72],[25.2,31.7]],np.float32)
    network = transform_points(points,matrix)
    decoded = heatmap_to_network(network_to_heatmap(network))
    np.testing.assert_allclose(transform_points(decoded,np.linalg.inv(matrix)),points,atol=2e-5)
    np.testing.assert_allclose(center_offsets([[70,36]],141,73),[[0,0]])


def test_local_decoder_subpixel():
    yy,xx = np.mgrid[:48,:48]
    expected = np.array([[12.2,23.3]])
    heat = -((xx-expected[0,0])**2+(yy-expected[0,1])**2)/(2*.8**2)
    xy,mass = decode_heatmaps(heat[None])
    np.testing.assert_allclose(xy,heatmap_to_network(expected),atol=.08)
    assert mass[0]>.95


def test_synthetic_determinism_and_real_label_roundtrip(tmp_path):
    data = SyntheticDataset(CONFIG,length=14,size=64,seed=12)
    image,target = data.sample(0)
    image2,target2 = data.sample(0)
    np.testing.assert_array_equal(image,image2)
    assert torch.equal(target['heatmaps'],target2['heatmaps'])
    assert torch.allclose(target['heatmaps'].sum((-1,-2)),target['visible'])
    spec = data.config['classes'][0]['points']
    write_image(tmp_path/'roi.png',image)
    record = {'image':'roi.png','class_id':0,'points':[{'id':p['id'],'xy':target['points'][i].tolist(),
               'visible':bool(target['visible'][i])} for i,p in enumerate(spec)]}
    (tmp_path/'data.jsonl').write_text(json.dumps(record))
    _,real = RealDataset(tmp_path/'data.jsonl',data.config,size=64)[0]
    torch.testing.assert_close(real['points'],target['points'])
    data.epoch=1
    assert not np.array_equal(image,data.sample(0)[0])


def test_heads_loss_gradients_and_all_negative():
    data = SyntheticDataset(CONFIG,length=7,size=64)
    x,t = data[0]
    model = ArmorNet().eval()
    outputs = model(torch.stack([x,x]))
    assert outputs[1].shape == (2,6,12,16,16)
    target = {k:torch.stack([v,v]) for k,v in t.items()}
    loss,_ = losses(outputs,target)
    loss.backward()
    gradient = model.landmarks[-1].weight.grad.reshape(6,12,64,1,1)
    assert gradient[0].abs().sum()>0
    assert gradient[1:].abs().sum()==0
    x,t = data[6]
    target = {k:v[None] for k,v in t.items()}
    loss,_ = losses(model(x[None]),target)
    assert torch.isfinite(loss)


def test_checkpoint_and_prediction_contract(tmp_path):
    data = SyntheticDataset(CONFIG,length=7,size=64)
    model = ArmorNet().eval()
    # Force class 0 to test decoding and inverse preprocessing, not random classification.
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.fill_(-10)
        model.classifier[-1].bias[0]=10
    ckpt = {'model':model.state_dict(),'config':data.config,'input_size':64,'status':'test_only'}
    path=tmp_path/'model.pt'
    torch.save(ckpt,path)
    restored,loaded=load_checkpoint(path)
    image=np.zeros((81,125,3),np.uint8)
    result=predict(restored,loaded,image,point_threshold=0,roi_to_frame=[[1,0,30],[0,1,40],[0,0,1]])
    assert len(result['points'])==len(data.config['classes'][0]['points'])
    for point in result['points']:
        np.testing.assert_allclose(point['offset_from_roi_center'],np.array(point['uv_roi'])-[62,40])
        np.testing.assert_allclose(point['uv_frame'],np.array(point['uv_roi'])+[30,40],atol=1e-5)
    assert result['checkpoint_status']=='test_only'
