#!/usr/bin/env python3
"""Check if UUID 1 has multiple persons merged by analyzing bbox clusters and embeddings."""
import json, paho.mqtt.client as mqtt, time, threading, base64, struct
import numpy as np
from collections import defaultdict

uuid_data = defaultdict(lambda: {'bboxes': [], 'embeddings': [], 'cameras': []})
lock = threading.Lock()

def decode_emb(b64):
    raw = base64.b64decode(b64)
    return np.array(struct.unpack(f'{len(raw)//4}f', raw))

def on_msg(client, userdata, msg):
    if '/data/' not in msg.topic:
        return
    try:
        data = json.loads(msg.payload)
        cam = data.get('id', '')
        persons = data.get('objects', {}).get('person', [])
        for p in persons:
            uid = str(p.get('id', ''))
            if not uid: continue
            bbox = p.get('bounding_box_px', {})
            meta = p.get('metadata')
            if not isinstance(meta, dict): continue
            reid = meta.get('reid')
            if not isinstance(reid, dict): continue
            emb_b64 = reid.get('embedding_vector', '')
            with lock:
                uuid_data[uid]['bboxes'].append({**bbox, 'cam': cam})
                uuid_data[uid]['cameras'].append(cam)
                if emb_b64 and len(uuid_data[uid]['embeddings']) < 30:
                    uuid_data[uid]['embeddings'].append({'emb': decode_emb(emb_b64), 'cam': cam, 'bbox': bbox})
    except:
        pass

client = mqtt.Client()
client.on_message = on_msg
client.connect('localhost', 1883)
client.subscribe('scenescape/data/camera/+')
client.loop_start()
print('Collecting data for 120s...')
time.sleep(120)
client.loop_stop()
client.disconnect()

uids = sorted(uuid_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
print(f'\nUnique UUIDs: {len(uids)}')

# For each UUID, check if bounding boxes cluster into multiple groups (=merged persons)
print(f'\n{"="*70}')
print(f'  BBOX CLUSTER ANALYSIS — does one UUID cover multiple positions?')
print(f'{"="*70}')
for uid in uids:
    for cam in sorted(set(uuid_data[uid]['cameras'])):
        bboxes = [b for b in uuid_data[uid]['bboxes'] if b.get('cam') == cam]
        if not bboxes: continue
        centers_x = [b['x'] + b.get('width',0)//2 for b in bboxes]
        centers_y = [b['y'] + b.get('height',0)//2 for b in bboxes]
        x_range = max(centers_x) - min(centers_x)
        y_range = max(centers_y) - min(centers_y)
        print(f'  UUID {uid} @ {cam}: {len(bboxes)} hits | center_x=[{min(centers_x)}-{max(centers_x)}] range={x_range}px | center_y=[{min(centers_y)}-{max(centers_y)}] range={y_range}px')
        if x_range > 400:
            print(f'    *** WIDE X RANGE — may contain multiple persons merged! ***')

# Check how many persons appear simultaneously in each frame
print(f'\n{"="*70}')
print(f'  SIMULTANEOUS DETECTIONS PER FRAME')
print(f'{"="*70}')
# Re-collect frame-level data
frame_counts = defaultdict(int)
# Count from uuid_data: how many UUIDs have bboxes
for uid in uids:
    cams = set(uuid_data[uid]['cameras'])
    print(f'  UUID {uid}: {len(uuid_data[uid]["bboxes"])} total detections across {sorted(cams)}')

# Pairwise embedding similarity
print(f'\n{"="*70}')
print(f'  PAIRWISE EMBEDDING SIMILARITY BETWEEN UUIDs')
print(f'{"="*70}')
uid_embs = {}
for uid in uids:
    embs = [e['emb'] for e in uuid_data[uid]['embeddings']]
    if embs:
        uid_embs[uid] = np.mean(embs, axis=0)

for i, u1 in enumerate(uids):
    for u2 in uids[i+1:]:
        if u1 in uid_embs and u2 in uid_embs:
            e1, e2 = uid_embs[u1], uid_embs[u2]
            cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            print(f'  UUID {u1} vs {u2}: cosine={cos:.4f}  {"<-- ABOVE 0.90 THRESHOLD (would merge!)" if cos > 0.90 else ""}')

# Intra-UUID embedding variance
print(f'\n{"="*70}')
print(f'  INTRA-UUID EMBEDDING SPREAD (high spread = multiple persons merged)')
print(f'{"="*70}')
for uid in uids:
    embs = [e['emb'] for e in uuid_data[uid]['embeddings']]
    if len(embs) < 2: continue
    dists = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            cos = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]))
            dists.append(cos)
    print(f'  UUID {uid}: {len(embs)} embeddings | min_cos={min(dists):.4f} avg_cos={np.mean(dists):.4f} max_cos={max(dists):.4f}')
    if min(dists) < 0.85:
        print(f'    *** LOW MIN COSINE — likely has multiple distinct persons merged! ***')

# Check per-camera embedding clusters within UUID 1
print(f'\n{"="*70}')
print(f'  UUID 1 DETAILED EMBEDDING ANALYSIS (per position)')
print(f'{"="*70}')
if '1' in uuid_data:
    emb_entries = uuid_data['1']['embeddings']
    for cam in sorted(set(e['cam'] for e in emb_entries)):
        cam_entries = [e for e in emb_entries if e['cam'] == cam]
        if len(cam_entries) < 2: continue
        # Group by x position
        for e in cam_entries:
            cx = e['bbox'].get('x',0) + e['bbox'].get('width',0)//2
            print(f'    {cam} x_center={cx} bbox={e["bbox"].get("width",0)}x{e["bbox"].get("height",0)}')

print(f'\n{"="*70}')
