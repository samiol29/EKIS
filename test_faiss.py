import faiss

idx = faiss.read_index("C:/Users/Mrith/OneDrive/Desktop/ekis-capstone/storage/faiss.index")
print("Index loaded. ntotal =", idx.ntotal)
try:
    print("Index dimension =", idx.d)
except:
    print("Could not read dimension")
