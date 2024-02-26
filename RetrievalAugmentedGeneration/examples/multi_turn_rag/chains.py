# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RAG example showcasing multi-turn conversation."""
import base64
import os
import logging
from pathlib import Path
from typing import Generator, List, Dict, Any

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnableAssign

# pylint: disable=no-name-in-module, disable=import-error
from RetrievalAugmentedGeneration.common.utils import (
    get_config,
    get_llm,
    get_vectorstore_langchain,
    get_embedding_model,
)
from RetrievalAugmentedGeneration.common.base import BaseExample
from operator import itemgetter

DOCS_DIR = os.path.abspath("./uploaded_files")
document_embedder = get_embedding_model()
docstore = None
settings = get_config()
logger = logging.getLogger(__name__)

class MultiTurnChatbot(BaseExample):

    def save_memory_and_get_output(self, d, vstore):
        """Accepts 'input'/'output' dictionary and saves to convstore"""
        vstore.add_texts(
            [
                f"User previously responded with {d.get('input')}",
                f"Agent previously responded with {d.get('output')}",
            ]
        )
        return d.get("output")

    def ingest_docs(self, file_name: str, filename: str):
        """Ingest documents to the VectorDB."""
        try:
            # TODO: Load embedding created in older conversation, memory persistance
            # We initialize class in every call therefore it should be global
            global docstore
            # Load raw documents from the directory
            # Data is copied to `DOCS_DIR` in common.server:upload_document
            _path = os.path.join(DOCS_DIR, filename)
            raw_documents = UnstructuredFileLoader(_path).load()

            if raw_documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.text_splitter.chunk_size,
                    chunk_overlap=settings.text_splitter.chunk_overlap,
                )
                documents = text_splitter.split_documents(raw_documents)
                if docstore:
                    docstore.add_documents(documents)
                else:
                    docstore = get_vectorstore_langchain(documents, document_embedder)
            else:
                logger.warning("No documents available to process!")
        except Exception as e:
            logger.error(f"Failed to ingest document due to exception {e}")
            raise ValueError(
                "Failed to upload document. Please upload an unstructured text document."
            )

    def llm_chain(
        self, context: str, question: str, num_tokens: str
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""

        logger.info("Using llm to generate response directly without knowledge base.")
        prompt_template = PromptTemplate.from_template(settings.prompts.chat_template)

        llm = get_llm()

        chain = prompt_template | llm | StrOutputParser()

        return chain.stream({"context_str": context, "query_str": question})

    def rag_chain(self, prompt: str, num_tokens: int) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above."""

        logger.info("Using rag to generate response from document")

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a document chatbot. Help the user as they ask questions about documents."
                    " User messaged just asked: {input}\n\n"
                    " From this, we have retrieved the following potentially-useful info: "
                    " Conversation History Retrieval:\n{history}\n\n"
                    " Document Retrieval:\n{context}\n\n"
                    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.)",
                ),
                ("user", "{input}"),
            ]
        )

        llm = get_llm()
        stream_chain = chat_prompt | llm | StrOutputParser()

        convstore = get_vectorstore_langchain(
            [], document_embedder, collection_name="conv_store"
        )

        resp_str = ""

        try:
            if docstore:
                retrieval_chain = (
                    RunnableAssign(
                        {"context": itemgetter("input") | docstore.as_retriever()}
                    )
                    | RunnableAssign(
                        {"history": itemgetter("input") | convstore.as_retriever()}
                    )
                )
                chain = retrieval_chain | stream_chain

                for chunk in chain.stream({"input": prompt}):
                    yield chunk
                    resp_str += chunk

                self.save_memory_and_get_output(
                    {"input": prompt, "output": resp_str}, convstore
                )

                return chain.stream(prompt)

        except Exception as e:
            logger.warning(f"Failed to generate response due to exception {e}")
        logger.warning(
            "No response generated from LLM, make sure you've ingested document."
        )
        return iter(
            [
                "No response generated from LLM, make sure you have ingested document from the Knowledge Base Tab."
            ]
        )

    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        """Search for the most relevant documents for the given search parameters."""

        try:
            if docstore != None:
                try:
                    retriever = docstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"score_threshold": 0.25},
                    )
                    docs = retriever.invoke(content)
                except NotImplementedError:
                    # Some retriever like milvus don't have similarity score threshold implemented
                    retriever = docstore.as_retriever()
                    docs = retriever.invoke(content)

                result = []
                for doc in docs:
                    result.append(
                        {
                            "source": os.path.basename(doc.metadata.get("source", "")),
                            "content": doc.page_content,
                        }
                    )
                return result
            return []
        except Exception as e:
            logger.error(f"Error from /documentSearch endpoint. Error details: {e}")
            return []
