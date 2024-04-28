# Copyright (c) Alibaba, Inc. and its affiliates.
from inference.models import api_model

from llmuses.models.model_adapter import ChatGenerationModelAdapter


class llama3_8b_instruct_origin(api_model):
    def __init__(self, workers=10):
        self.temperature = 0.95
        self.max_tokens = 1024
        self.workers = workers

        self.model_adapter = ChatGenerationModelAdapter(model_id='LLM-Research/Meta-Llama-3-8B-Instruct',
                                                        model_revision='master',
                                                        template_type='llama3')

        super().__init__(workers)

    def get_api_result(self, sample):
        question = sample["question"]

        res_d: dict = self.model_adapter.predict(question,
                                                 infer_cfg={
                                                     'do_sample': True,
                                                     'max_new_tokens': 128,
                                                     'temperature': 0.95,
                                                            }
                                                 )

        ans: str = res_d['choices'][0]['message']['content']

        return ans
