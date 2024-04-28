# Copyright (c) Alibaba, Inc. and its affiliates.
from inference.models import api_model

from llmuses.models.model_adapter import ChatGenerationModelAdapter


class llama3_chinese_sft_v3(api_model):
    def __init__(self, workers=10):
        self.temperature = 0.95
        self.max_tokens = 1024
        self.workers = workers

        self.model_adapter = ChatGenerationModelAdapter(model_id='zhuangxialie/Llama3_Chinese_Sft',
                                                        model_revision='master',
                                                        template_type='llama3')

        super().__init__(workers)

    def get_api_result(self, sample):
        question = sample["question"]

        res_d: dict = self.model_adapter.predict(question,
                                                 infer_cfg={'max_new_tokens': 128,
                                                            'temperature': 0.95,
                                                            }
                                                 )
        ans: str = res_d['choices'][0]['message']['content']

        return ans
