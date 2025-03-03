import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt

chroma_client = chromadb.PersistentClient("./data/chroma.db")


data_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()


collection = chroma_client.get_or_create_collection(
    "multimodel_collection",
    embedding_function=embedding_function,
    data_loader=data_loader
)

# collection.update(
#     ids=["1", "2"],
#     uris=["./images/tiger.jpg", "./images/lion.jpg"],
#     metadatas=[{"category": "animal"}, {"category": "animal"}]
# )



def print_query_results(query_list:list, query_results:dict) -> None:
    result_count = len(query_results["ids"][0])
    for i in range(len(query_list)):
        print(f"Query : {query_list[i]}")
        
        for j in range(result_count):
            id = query_results["ids"][i][j]
            distance = query_results["distances"][i][j]
            data = query_results["data"][i][j]
            document = query_results["documents"][i][j]
            metadata = query_results["metadatas"][i][j]
            uri = query_results["uris"][i][j]

            print(f"id: {id}, distance: {distance}, data: {data}, document: {document}, metadata: {metadata}, uri: {uri}")


            print(f"data: {uri}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()


query_texts = ["tiger"]

query_results = collection.query(
    query_texts=query_texts,
    n_results=3,
    include=["data", "documents", "metadatas", "uris", "distances"]
)

print_query_results(query_list=query_texts, query_results=query_results)