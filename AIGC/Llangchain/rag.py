#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   rag.py
@Time   :   2024/11/08 10:09:49
@Author :   yhao 
@Email  :   data4.cmi@unilever.com
@Desc   :   学习langchain的RAG
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

from typing import Any, List
from pydantic import BaseModel, Field, ConfigDict
import torch
import bs4
import langchain_core
from langchain_community.document_loaders import WebBaseLoader, AzureBlobStorageContainerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


def indexing_load():
    """加载数据源
    """
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    # loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs={"parse_only": bs4_strainer},
    # )
    loader = AzureBlobStorageContainerLoader(
        conn_str="sampleconnectionstring",
        container="azureml-blobstore-de30a0c3-5106-4dd5-b8ee-32f7c565ec7f",
        prefix="RAG"
    )
    docs: List[langchain_core.documens.base.Document] = loader.load()

    print(len(docs[0].page_content))

    return docs


def indexing_load_azure_pdf():
    pass


def indexing_split(docs):
    """将文档切块
    1、文档太长，模型放不下
    2、检索时只返回最有关的块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        add_start_index=True # restore start_index in metadata 
    )
    all_splits = text_splitter.split_documents(docs)
    
    return all_splits


def indexing_store(all_splits):
    """这个好像是要api key的
    """
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    return vectorstore


class CustomLlamaEmbedding(BaseModel, Embeddings):
    """使用Llama3.1的embedding
    """
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    # tokenizer: Any
    context_sequence_length: int = 512
    query_sequence_length: int = 512
    model_name: str = ''
    """Model name to use."""

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, protected_namespaces=()
    )

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer.
        """
        super().__init__(**kwargs)  # 用了BaseModel且定义了__init__就要super
        self.client = SentenceTransformer(
            "/media/data/LLMS/Llama3-hf",
            device="cpu",
            trust_remote_code=True
        )
        # 设置llama tokenizer pad_token
        self.client.tokenizer.pad_token = self.client.tokenizer.eos_token
        self.context_sequence_length = 512
        self.query_sequence_length = 512

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with torch.no_grad():
            embeddings = self.client.encode(texts)
            embeddings = embeddings.astype("float32")
            return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """First element of model_output contains all token embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        result = torch.sum(token_embeddings * input_mask_expanded, 1) / \
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return result
        

def indexing_store_local_llama(all_splits):
    """使用本地的llama3.1 tokenizer
    """
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=CustomLlamaEmbedding(),
        persist_directory="/home/yhao/temp/chroma"
    )
    return vectorstore


def RagChat():
    """Put it all together into a chain that 
    takes a question, retrieves relevant documents, constructs a prompt,
    passes it into a model, and parses the output
    """
    pass


if __name__ == "__main__":
    docs = indexing_load()
    all_splits = indexing_split(docs)
    vectorstore = indexing_store_local_llama(all_splits)
    print()