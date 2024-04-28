# Copyright (c) Alibaba, Inc. and its affiliates.
from inference.models import api_model

from llmuses.models.model_adapter import ChatGenerationModelAdapter


class Llama3_8B_Instruct_Origin(api_model):
    def __init__(self, workers=10):
        self.temperature = 0.95
        self.max_tokens = 1024
        self.workers = workers

        self.model_adapter = ChatGenerationModelAdapter(model_id='LLM-Research/Meta-Llama-3-8B-Instruct',
                                                        model_revision='master',)

        super().__init__(workers)

    def get_api_result(self, sample):
        question = sample["question"]

        res_d: dict = self.model_adapter.predict(question)
        ans: str = res_d['choices'][0]['message']['content']

        return ans
