#!/usr/bin/env python3
"""Analyze reid embeddings from MQTT to check if UUIDs represent same or different persons."""
import json
import paho.mqtt.client as mqtt
import time
import threading
import base64
import struct
import numpy as np
from collections import defaultdict

embeddings = defaultdict(list)
lock = threading.Lock()

def decode_emb(b64):
    raw = base64.b64decode(b64)
    n = len(raw) // 4
    return np.array(struct.unpack(f'{n}f', raw))

def on_msg(client, userdata, msg):
    if '/data/' not in msg.topic:
        return
    try:
        data = json.loads(msg.payload)
        persons = data.get('objects', {}).get('person', [])
        for p in persons:
            uid = str(p.get('id', ''))
            if not uid:
                continue
            meta = p.get('metadata')
            if not isinstance(meta, dict):
                continue
            reid = meta.get('reid')
            if not isinstance(reid, dict):
                continue
            emb_b64 = reid.get('embedding_vector', '')
            if not emb_b64:
                continue
            emb = decode_emb(emb_b64)
            with lock:
                if len(embeddings[uid]) < 10:
                    embeddings[uid].append(emb)
    except Exception as e:
        pass  # silently skip malformed messages

client = mqtt.Client()
client.on_message = on_msg
client.connect('localhost', 1883)
client.subscribe('scenescape/data/camera/+')
client.loop_start()
print('Collecting embeddings for 60s...')
time.sleep(60)
client.loop_stop()
client.disconnect()

print(f'\nUUIDs with embeddings: {len(embeddings)}')
for uid in sorted(embeddings, key=lambda x: int(x) if x.isdigit() else 0):
    embs = embeddings[uid]
    print(f'  UUID {uid}: {len(embs)} embeddings, dim={embs[0].shape[0]}')

print('\n=== L2 distances between UUID embeddings ===')
uids = sorted(embeddings, key=lambda x: int(x) if x.isdigit() else 0)
for i, u1 in enumerate(uids):
    for u2 in uids[i+1:]:
        e1 = np.mean(embeddings[u1], axis=0)
        e2 = np.mean(embeddings[u2], axis=0)
        l2 = np.linalg.norm(e1 - e2)
        cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        print(f'  UUID {u1} vs {u2}: L2={l2:.2f}, cosine_sim={cos:.4f}')

print('\n=== Intra-UUID L2 distances (self-consistency) ===')
for uid in uids:
    if len(embeddings[uid]) < 2:
        print(f'  UUID {uid}: only 1 embedding, skipping')
        continue
    dists = []
    for i in range(len(embeddings[uid])):
        for j in range(i+1, len(embeddings[uid])):
            dists.append(np.linalg.norm(embeddings[uid][i] - embeddings[uid][j]))
    print(f'  UUID {uid}: avg_self_dist={np.mean(dists):.2f}, max={np.max(dists):.2f}')

# Also check which model is being used
print('\n=== Model info ===')
def on_model_check(client, userdata, msg):
    if '/data/' not in msg.topic:
        return
    data = json.loads(msg.payload)
    persons = data.get('objects', {}).get('person', [])
    for p in persons:
        meta = p.get('metadata', {})
        if isinstance(meta, dict):
            reid = meta.get('reid', {})
            if isinstance(reid, dict):
                print(f'  reid model_name: {reid.get("model_name", "N/A")}')
                emb = reid.get('embedding_vector', '')
                if emb:
                    raw = base64.b64decode(emb)
                    print(f'  embedding bytes: {len(raw)}, floats: {len(raw)//4}')
        client.disconnect()
        return

client2 = mqtt.Client()
client2.on_message = on_model_check
client2.connect('localhost', 1883)
client2.subscribe('scenescape/data/camera/+')
client2.loop_forever()
