# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The definition of the Llama Index chain server."""
import base64
import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List
import importlib
from inspect import getmembers, isclass

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymilvus.exceptions import MilvusException, MilvusUnavailableException
from RetrievalAugmentedGeneration.common import utils, tracing


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create the FastAPI server
app = FastAPI()

origins = [
    "http://localhost:3001",
    "http://localhost:6006",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXAMPLE_DIR = "RetrievalAugmentedGeneration/example"

class Message(BaseModel):
    """Definition of the Chat Message type."""
    role: str = Field(description="Role for a message AI, User and System")
    content: str = Field(description="The input query/prompt to the pipeline.")

    @validator('role')
    def validate_role(cls, value):
        valid_roles = {'user', 'assistant', 'system'}
        if value.lower() not in valid_roles:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
        return value.lower()

class Prompt(BaseModel):
    """Definition of the Prompt API data type."""
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far. The roles of the messages must be alternating between user and assistant. The last input message should have role user. A message with the the system role is optional, and must be the very first message if it is present.")
    use_knowledge_base: bool = Field(..., description="Whether to use a knowledge base")
    temperature: float = Field(0.2, description="The sampling temperature to use for text generation. The higher the temperature value is, the less deterministic the output text will be. It is not recommended to modify both temperature and top_p in the same call.")
    top_p: float = Field(0.7, description="The top-p sampling mass used for text generation. The top-p value determines the probability mass that is sampled at sampling time. For example, if top_p = 0.2, only the most likely tokens (summing to 0.2 cumulative probability) will be sampled. It is not recommended to modify both temperature and top_p in the same call.")
    max_tokens: int = Field(1024, description="The maximum number of tokens to generate in any given call. Note that the model is not aware of this value, and generation will simply stop at the number of tokens specified.")
    seed: int = Field(42, description="If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result.")
    bad: List[str] = Field(None, description="A word or list of words not to use. The words are case sensitive.")
    stop: List[str] = Field(None, description="A string or a list of strings where the API will stop generating further tokens. The returned text will not contain the stop sequence.")
    stream: bool = Field(False, description="If set, partial message deltas will be sent. Tokens will be sent as data-only server-sent events (SSE) as they become available (JSON responses are prefixed by data:), with the stream terminated by a data: [DONE] message.")


class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    content: str = Field(description="The content or keywords to search for within documents.")
    num_docs: int = Field(description="The maximum number of documents to return in the response.", default=4)


@app.on_event("startup")
def import_example() -> None:
    """
    Import the example class from the specified example file.
    The example directory is expected to have a python file where the example class is defined.
    """

    for root, dirs, files in os.walk(EXAMPLE_DIR):
        for file in files:
            if not file.endswith(".py"):
                continue

            # Import the specified file dynamically
            spec = importlib.util.spec_from_file_location(name="example", location=os.path.join(root, file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Scan each class in the file to find one with the 3 implemented methods: ingest_docs, rag_chain and llm_chain
            for name, _ in getmembers(module, isclass):
                try:
                    cls = getattr(module, name)
                    if set(["ingest_docs", "llm_chain", "rag_chain"]).issubset(set(dir(cls))):
                        if name == "BaseExample":
                            continue
                        example = cls()
                        app.example = cls
                        return
                except:
                    raise ValueError(f"Class {name} is not implemented and could not be instantiated.")

    raise NotImplementedError(f"Could not find a valid example class in {EXAMPLE_DIR}")


@app.post("/uploadDocument")
@tracing.instrumentation_wrapper
async def upload_document(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document to the vector store."""
    if not file.filename:
        return JSONResponse(content={"message": "No files provided"}, status_code=200)

    try:
        upload_folder = "uploaded_files"
        upload_file = os.path.basename(file.filename)
        if not upload_file:
            raise RuntimeError("Error parsing uploaded filename.")
        file_path = os.path.join(upload_folder, upload_file)
        uploads_dir = Path(upload_folder)
        uploads_dir.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        app.example().ingest_docs(file_path, upload_file)

        return JSONResponse(
            content={"message": "File uploaded successfully"}, status_code=200
        )

    except Exception as e:
        logger.error("Error from /uploadDocument endpoint. Ingestion of file: " + file.filename + " failed with error: " + str(e))
        return JSONResponse(
            content={"message": str(e)}, status_code=500
        )


@app.post("/generate")
@tracing.instrumentation_wrapper
async def generate_answer(request: Request, prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""
    
    chat_history = prompt.messages
    # The last user message will be the query for the rag or llm chain
    last_user_message = next((message.content for message in reversed(chat_history) if message.role == 'user'), None)
    
    # Find and remove the last user message if present
    for i in reversed(range(len(chat_history))):
        if chat_history[i].role == 'user':
            del chat_history[i]
            break  # Remove only the last user message
    
    # All the other information from the prompt like the temperature, top_p etc., are llm_settings
    llm_settings =  {
            key: value
            for key, value in vars(prompt).items()
            if key not in ['messages', 'use_knowledge_base']
        }
    try:
        example = app.example()
        if prompt.use_knowledge_base:
            logger.info("Knowledge base is enabled. Using rag chain for response generation.")
            generator = example.rag_chain(query=last_user_message, chat_history=chat_history, **llm_settings)
            return StreamingResponse(generator, media_type="text/event-stream")

        generator = example.llm_chain(query=last_user_message, chat_history=chat_history, **llm_settings)
        return StreamingResponse(generator, media_type="text/event-stream")

    except (MilvusException, MilvusUnavailableException) as e:
        logger.error(f"Error from Milvus database in /generate endpoint. Please ensure you have ingested some documents. Error details: {e}")
        return StreamingResponse(iter(["Error from milvus server. Please ensure you have ingested some documents. Please check chain-server logs for more details."]), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error from /generate endpoint. Error details: {e}")
        return StreamingResponse(iter(["Error from chain server. Please check chain-server logs for more details."]), media_type="text/event-stream")


@app.post("/documentSearch")
@tracing.instrumentation_wrapper
async def document_search(request: Request,data: DocumentSearch) -> List[Dict[str, Any]]:
    """Search for the most relevant documents for the given search parameters."""

    try:
        example = app.example()
        if hasattr(example, "document_search") and callable(example.document_search):
            return example.document_search(data.content, data.num_docs)

        raise NotImplementedError("Example class has not implemented the document_search method.")

    except Exception as e:
        logger.error(f"Error from /documentSearch endpoint. Error details: {e}")
        return []