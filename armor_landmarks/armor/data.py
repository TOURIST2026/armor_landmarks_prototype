import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import letterbox, transform_points, network_to_heatmap

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    buffer.tofile(str(path))


def load_templates(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    classes = config['classes']
    if [c['id'] for c in classes] != list(range(len(classes))):
        raise ValueError('Class IDs must be contiguous, starting at zero')
    for cls in classes:
        points = cls['points']
        if not 4 <= len(points) <= config['max_points']:
            raise ValueError('Each class requires 4..max_points points')
        if len({p['id'] for p in points}) != len(points):
            raise ValueError('Duplicate point IDs')
        xy = np.array([p['xy'] for p in points], np.float32)
        image = read_image(path.parent / cls['image'])
        h, w = image.shape[:2]
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= [w, h]):
            raise ValueError(f'Out-of-bounds annotation: {cls["name"]}')
        if np.linalg.matrix_rank(xy - xy.mean(0)) < 2:
            raise ValueError('Collinear keypoints')
    return config


def image_tensor(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def targets(class_id, points, visible, count, max_points, size):
    heat_size = size // 4
    yy, xx = np.mgrid[:heat_size, :heat_size]
    heatmaps = np.zeros((max_points, heat_size, heat_size), np.float32)
    coords = np.zeros((max_points, 2), np.float32)
    exists = np.zeros(max_points, np.float32)
    vis = np.zeros(max_points, np.float32)
    if count:
        coords[:count] = points
        exists[:count] = 1
        vis[:count] = visible
        hp = network_to_heatmap(points)
        for i, (x, y) in enumerate(hp):
            if visible[i]:
                values = np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * 1.2**2))
                heatmaps[i] = values / max(values.sum(), 1e-12)
    return {
        'class_id': torch.tensor(class_id, dtype=torch.long),
        'heatmaps': torch.from_numpy(heatmaps), 'points': torch.from_numpy(coords),
        'exists': torch.from_numpy(exists), 'visible': torch.from_numpy(vis),
        'roi_scale': torch.ones(2,dtype=torch.float32),
    }


class SyntheticDataset(Dataset):
    """Deterministic per (seed, epoch, index); validation uses disjoint seed."""
    def __init__(self, config_path, length=2048, size=192, seed=123, augment=True):
        self.config_path = Path(config_path).resolve()
        self.config = load_templates(config_path)
        self.images = [read_image(self.config_path.parent / c['image']) for c in self.config['classes']]
        self.length, self.size, self.seed, self.epoch = length, size, seed, 0
        self.augment = augment

    def __len__(self):
        return self.length

    def sample(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        size = self.size
        nclasses = len(self.images)
        class_id = index % (nclasses + 1)
        background = rng.uniform(15, 160, (size, size, 3)).astype(np.uint8)
        background = cv2.GaussianBlur(background, (15, 15), 0)
        if class_id == nclasses:
            for _ in range(12):
                a, b = rng.integers(0, size, (2, 2))
                color = tuple(map(int, rng.integers(0, 256, 3)))
                cv2.line(background, tuple(a), tuple(b), color, int(rng.integers(1, 7)))
            return background, targets(class_id, [], [], 0, self.config['max_points'], size)
        template = self.images[class_id]
        h, w = template.shape[:2]
        source = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        scale = rng.uniform(0.55, 0.95) * size / max(h, w)
        angle = rng.uniform(-35, 35) * np.pi / 180
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dest = (source - [(w-1)/2, (h-1)/2]) @ rotation.T * scale
        dest += (size-1)/2 + rng.uniform(-0.12, 0.12, 2) * size
        dest += rng.uniform(-0.07, 0.07, (4, 2)) * size
        matrix = cv2.getPerspectiveTransform(source, dest.astype(np.float32))
        warped = cv2.warpPerspective(template, matrix, (size, size))
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (size, size))
        alpha = mask[..., None].astype(np.float32) / 255
        image = (warped.astype(np.float32) + background * (1-alpha)).clip(0,255).astype(np.uint8)
        raw_points = [p['xy'] for p in self.config['classes'][class_id]['points']]
        points = transform_points(raw_points, matrix)
        visible = np.all((points >= 0) & (points <= size-1), axis=1)
        if self.augment:
            if rng.random() < 0.3:
                x, y = rng.integers(0, size, 2)
                ow, oh = rng.integers(size//12, size//3, 2)
                image[y:y+oh, x:x+ow] = rng.integers(0, 256, 3)
                visible &= ~((points[:,0]>=x)&(points[:,0]<x+ow)&(points[:,1]>=y)&(points[:,1]<y+oh))
            small = int(rng.integers(max(24, size//5), size+1))
            image = cv2.resize(cv2.resize(image, (small, small), interpolation=cv2.INTER_AREA), (size,size))
            if rng.random() < 0.5:
                image = cv2.GaussianBlur(image, (3,3), float(rng.uniform(0.3,1.4)))
            if rng.random() < 0.25:
                kernel = np.zeros((5,5),np.float32)
                kernel[2,:] = 0.2
                image = cv2.filter2D(image,-1,kernel)
            image = image.astype(np.float32)*rng.uniform(0.65,1.35)+rng.uniform(-20,20)
            image += rng.normal(0,rng.uniform(0,8),image.shape)
            image = image.clip(0,255).astype(np.uint8)
            if rng.random() < 0.3:
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return image, targets(class_id, points, visible, len(points), self.config['max_points'], size)

    def __getitem__(self, index):
        image, target = self.sample(index)
        return image_tensor(image), target


class RealDataset(Dataset):
    """JSONL records: image, class_id, points=[{id,xy,visible}]. xy is raw ROI pixels."""
    def __init__(self, manifest, config, size=192):
        self.path = Path(manifest).resolve()
        self.records = [json.loads(x) for x in self.path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        self.config, self.size = config, size
        if not self.records:
            raise ValueError('Empty real-data manifest')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_image(self.path.parent / record['image'])
        net_image, matrix = letterbox(image, self.size)
        cid = record['class_id']
        if not 0 <= cid <= len(self.config['classes']):
            raise ValueError('Invalid real-data class ID')
        if cid == len(self.config['classes']):
            target = targets(cid, [], [], 0, self.config['max_points'], self.size)
            target['roi_scale'] = torch.tensor([matrix[0,0],matrix[1,1]],dtype=torch.float32)
            return image_tensor(net_image), target
        spec = self.config['classes'][cid]['points']
        annotations = {p['id']: p for p in record['points']}
        xy = np.array([annotations[p['id']]['xy'] for p in spec], np.float32)
        visible = np.array([annotations[p['id']]['visible'] for p in spec], bool)
        h,w = image.shape[:2]
        visible &= np.all((xy >= 0) & (xy <= [w-1,h-1]),axis=1)
        points = transform_points(xy, matrix)
        target = targets(cid, points, visible, len(spec), self.config['max_points'], self.size)
        target['roi_scale'] = torch.tensor([matrix[0,0],matrix[1,1]],dtype=torch.float32)
        return image_tensor(net_image), target
