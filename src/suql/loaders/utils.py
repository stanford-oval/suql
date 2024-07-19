import json

def chunk_store_documents(data, output_file): 
    from llama_index.core.schema import Document
    data = [Document(text=data)] # llama index expects a list
    
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")

    from llama_index.core.node_parser import SemanticSplitterNodeParser
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(data)
    
    chunked_documents = [node.text for node in nodes]
    
    with open(output_file, "w") as fd:
        json.dump(
            chunked_documents, 
            fd,
            indent=2
        )