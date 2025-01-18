import unitylib

builder = unitylib.SearcherCreator("hnsw")
builder.set("hnswlib_index_path",
            "/data/home/petrizhang/develop/unity/examples/hnswlib.bin")
builder.set("dim", 256) 
builder.set("metric", "L2") 
searcher = builder.build()

print(searcher)
