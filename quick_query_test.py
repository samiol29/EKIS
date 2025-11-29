import numpy as np, faiss, json
idx = faiss.read_index("storage/faiss.index")
idmap = json.load(open("storage/id_map.json"))
# use mean embedding as a synthetic query
q = np.mean(np.load("storage/embeddings.npy"), axis=0).astype("float32")
q = q / (np.linalg.norm(q)+1e-12)
D,I = idx.search(q.reshape(1,-1), 5)
print("indices:", I.tolist(), "scores:", D.tolist())

# safe mapping that ignores -1
mapped = []
for ind in I[0]:
    if int(ind) < 0:
        mapped.append(None)
    else:
        if isinstance(idmap, dict):
            mapped.append(idmap.get(str(int(ind)), f"missing_map_for_{ind}"))
        else:
            mapped.append(idmap[int(ind)] if int(ind) < len(idmap) else f"out_of_range_{ind}")
print("mapped:", mapped)
