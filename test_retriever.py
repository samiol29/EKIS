from services.storage.retriever import retrieve

print("\n=== BASIC QUERY TEST ===")
res = retrieve("test query", k=5)
print(res)

print("\n=== FILTER BY FILENAME ===")
res = retrieve("test", filename="your_file.txt")
print(res)

print("\n=== FILTER BY FILETYPE ===")
res = retrieve("test", filetype=".txt")
print(res)

print("\n=== FILTER BY MIN SCORE ===")
res = retrieve("test", min_score=0.5)
print(res)

print("\n=== COMBINED FILTER TEST ===")
res = retrieve("test", filetype=".txt", min_score=0.3, max_results=2)
print(res)

print("\n=== FILTER TEST: min_score 0.5 ===")
print(retrieve("waterfall model", filetype=".pdf", min_score=0.5))

print("\n=== FILTER TEST: min_score 0.2 ===")
print(retrieve("waterfall", filetype=".pdf", min_score=0.2))
