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

"""LLM Chains for executing Retrival Augmented Generation."""
import logging
from typing import Generator

import pandas as pd
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from pandasai import Agent as PandasAI_Agent
from pandasai.responses.response_parser import ResponseParser

from integrations.pandasai.llms.nv_aiplay import NVIDIA as PandasAI_NVIDIA
from RetrievalAugmentedGeneration.common.base import BaseExample
from RetrievalAugmentedGeneration.common.utils import (
    get_config,
    get_llm,
    set_service_context,
)

# pylint: disable=no-name-in-module, disable=import-error
from RetrievalAugmentedGeneration.example.csv_utils import (
    extract_df_desc,
    get_prompt_params,
    parse_prompt_config,
)

logger = logging.getLogger(__name__)
settings = get_config()

FOUNDATIONAL_LLM_PD = "playground_nv_llama2_rlhf_70b"


class PandasDataFrame(ResponseParser):
    """Returns Pandas Dataframe instead of SmartDataFrame"""

    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        return result["value"]


class CSVChatbot(BaseExample):
    """RAG example showcasing CSV parsing using Pandas AI Agent"""

    def read_and_concatenate_csv(self, file_paths_txt):
        """Reads CSVs and concatenates their data"""

        with open(file_paths_txt, "r", encoding="UTF-8") as file:
            file_paths = file.read().splitlines()

        concatenated_df = pd.DataFrame()
        reference_columns = None

        for i, path in enumerate(file_paths):
            df = pd.read_csv(path)

            if i == 0:
                reference_columns = df.columns
                concatenated_df = df
            else:
                if not df.columns.equals(reference_columns):
                    raise ValueError(
                        f"Columns of the file {path} do not match the reference columns."
                    )
                concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

        return concatenated_df

    def ingest_docs(self, data_dir: str, filename: str):
        """Ingest documents to the VectorDB."""

        with open("ingested_csv_files.txt", "a", encoding="UTF-8") as f:
            f.write(data_dir + "\n")

        logger.info("Document %s ingested successfully", filename)

    def llm_chain(
        self, context: str, question: str, num_tokens: int
    ) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""

        logger.info("Using llm to generate response directly without knowledge base.")
        prompt = ChatPromptTemplate.from_messages(
            [("system", get_config().prompts.chat_template), ("user", "{input}")]
        )

        logger.info("Using prompt for response: %s", prompt)

        chain = prompt | get_llm() | StrOutputParser()
        return chain.stream({"input": question})

    def rag_chain(self, prompt: str, num_tokens: int) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above."""

        logger.info("Using rag to generate response from document")
        llm = get_llm()

        df = self.read_and_concatenate_csv(file_paths_txt="ingested_csv_files.txt")
        df = df.fillna(0)

        df_desc = extract_df_desc(df)
        prompt_config = parse_prompt_config(
            "RetrievalAugmentedGeneration/example/csv_prompt_config.yaml"
        )

        logger.info(prompt_config.get("csv_prompts", []))
        data_retrieval_prompt_params = get_prompt_params(
            prompt_config.get("csv_prompts", [])
        )
        llm_data_retrieval = PandasAI_NVIDIA(temperature=0.2, model=FOUNDATIONAL_LLM_PD)

        config_data_retrieval = {
            "llm": llm_data_retrieval,
            "response_parser": PandasDataFrame,
        }
        agent_data_retrieval = PandasAI_Agent(
            [df], config=config_data_retrieval, memory_size=20
        )

        data_retrieval_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    prompt_config.get("csv_data_retrieval_template", [])
                ),
                HumanMessagePromptTemplate.from_template("{query}"),
            ],
            input_variables=["description", "instructions", "data_frame", "query"],
        )

        result_df = agent_data_retrieval.chat(
            data_retrieval_prompt.format_prompt(
                description=data_retrieval_prompt_params.get("description"),
                instructions=data_retrieval_prompt_params.get("instructions"),
                data_frame=df_desc,
                query=prompt,
            ).to_string()
        )
        logger.info("Result Data Frame: %s", result_df)

        response_prompt_template = PromptTemplate(
            template=prompt_config.get("csv_response_template", []),
            input_variables=["query", "data"],
        )
        response_prompt = response_prompt_template.format(query=prompt, data=result_df)

        logger.info("Using prompt for response: %s", response_prompt)

        chain = response_prompt_template | llm | StrOutputParser()
        return chain.stream({"query": prompt, "data": result_df})
