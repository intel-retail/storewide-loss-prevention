#!/usr/bin/env python3
"""Check if UUIDs represent distinct persons or the same person fragmented."""
import json, paho.mqtt.client as mqtt, time, threading, base64, struct
import numpy as np
from collections import defaultdict

cooccurrence = defaultdict(set)
uuid_bboxes = defaultdict(list)
uuid_embeddings = defaultdict(list)
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
        ts = data.get('timestamp', '')
        persons = data.get('objects', {}).get('person', [])
        frame_uids = []
        for p in persons:
            uid = str(p.get('id', ''))
            if not uid: continue
            frame_uids.append(uid)
            bbox = p.get('bounding_box_px', {})
            meta = p.get('metadata')
            if not isinstance(meta, dict): continue
            reid = meta.get('reid')
            if not isinstance(reid, dict): continue
            emb_b64 = reid.get('embedding_vector', '')
            with lock:
                uuid_bboxes[uid].append({'cam': cam, 'bbox': bbox, 'ts': ts})
                if emb_b64 and len(uuid_embeddings[uid]) < 5:
                    uuid_embeddings[uid].append(decode_emb(emb_b64))
        with lock:
            cooccurrence[f"{cam}_{ts}"].update(frame_uids)
    except:
        pass

client = mqtt.Client()
client.on_message = on_msg
client.connect('localhost', 1883)
client.subscribe('scenescape/data/camera/+')
client.loop_start()
import sys
dur = int(sys.argv[1]) if len(sys.argv) > 1 else 120
print(f'Collecting data for {dur}s...')
time.sleep(dur)
client.loop_stop()
client.disconnect()

uids = sorted(uuid_bboxes.keys(), key=lambda x: int(x) if x.isdigit() else 0)
print(f'\nUnique UUIDs: {len(uids)}')
print(f'Total frames: {len(cooccurrence)}')

print(f'\n{"="*70}')
print(f'  CO-OCCURRENCE: Do UUIDs appear in the same frame?')
print(f'{"="*70}')
for i, u1 in enumerate(uids):
    for u2 in uids[i+1:]:
        same = sum(1 for v in cooccurrence.values() if u1 in v and u2 in v)
        if same > 0:
            print(f'  UUID {u1} & {u2}: co-occur in {same} frames => DIFFERENT PERSONS')
        else:
            print(f'  UUID {u1} & {u2}: NEVER co-occur => COULD BE SAME PERSON')

print(f'\n{"="*70}')
print(f'  BBOX POSITIONS')
print(f'{"="*70}')
for uid in uids:
    for cam in sorted(set(e['cam'] for e in uuid_bboxes[uid])):
        bboxes = [e['bbox'] for e in uuid_bboxes[uid] if e['cam'] == cam]
        xs = [b.get('x',0) for b in bboxes]
        ys = [b.get('y',0) for b in bboxes]
        ws = [b.get('width',0) for b in bboxes]
        hs = [b.get('height',0) for b in bboxes]
        print(f'  UUID {uid} @ {cam}: {len(bboxes)} hits | x=[{min(xs)}-{max(xs)}] y=[{min(ys)}-{max(ys)}] size=[{min(ws)}x{min(hs)}-{max(ws)}x{max(hs)}]')

print(f'\n{"="*70}')
print(f'  REID EMBEDDING SIMILARITY')
print(f'{"="*70}')
uid_embs = {}
for uid in uids:
    if uuid_embeddings[uid]:
        uid_embs[uid] = np.mean(uuid_embeddings[uid], axis=0)
for i, u1 in enumerate(uids):
    for u2 in uids[i+1:]:
        if u1 in uid_embs and u2 in uid_embs:
            e1, e2 = uid_embs[u1], uid_embs[u2]
            cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            verdict = "SAME PERSON!" if cos > 0.95 else "LIKELY SAME" if cos > 0.90 else "BORDERLINE" if cos > 0.80 else "DIFFERENT"
            print(f'  UUID {u1} vs {u2}: cosine={cos:.4f} => {verdict}')

print(f'\n{"="*70}')
print(f'  TEMPORAL ANALYSIS')
print(f'{"="*70}')
for uid in uids:
    ts_list = sorted(e['ts'] for e in uuid_bboxes[uid])
    print(f'  UUID {uid}: first={ts_list[0]}, last={ts_list[-1]}, count={len(ts_list)}')
print(f'{"="*70}')
