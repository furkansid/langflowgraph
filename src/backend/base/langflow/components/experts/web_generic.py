from pathlib import Path

from langflow.base.io.text import TextComponent
from langflow.base.data.utils import TEXT_FILE_TYPES, parse_text_file_to_data
from langflow.custom import Component
from langflow.io import Output, StateInput, StrInput, MessageTextInput
from langflow.schema.data import State
from langflow.base.experts_base.generic_web import WebLoader



class WebGeneric(Component):
    display_name = "Generic Web Expert"
    description = "Expert in web which analyze and provide pages information"
    icon = "globe"
    name = "generic-web-expert"
    SUPPORTED_FILE_TYPES = ["pdf"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    inputs = [
         MessageTextInput(
            name='test_question',
            display_name='Test Question',
            required=True,
            advanced=True,
            show=False
        ),
        StateInput(
            name='state',
            display_name='from',
            info="Link to state or logic box",
            is_list=True
            
        ),
        StrInput(
            name="urls",
            display_name="URLs",
            list=True,
            required=True,
        )
        ]

    outputs = [
        Output(display_name="pages", name="data", method="dry_run"),
    ]



    async def web_retriever(self):
        urls = self.urls
        if not urls:
            raise ValueError("Please, provide atleast one url.")

        retriever = await WebLoader(urls, ollama_embedder_model='mxbai-embed-large').retreiver(num_of_docs=2)
        return retriever
    
    async def dry_run(self) -> State:
        try:
            previous_state = self.state[0].data
        except AttributeError:
            return None
        if 'question' not in previous_state:
            if not self.test_question:
                raise ValueError("Please, provide a question to use this component.")
            test_question = self.test_question
        else:
            test_question = previous_state['question']
        state = {
            'question': test_question
        }
        retriever = await self.web_retriever(state)
        docs = await retriever.ainvoke(test_question)
        results = {**previous_state, **{'documents': docs}}
        return State(data=results, text_key='documents')


    async def langgraph_prepare(self):
        self.retriever = await self.web_retriever()

    async def langgraph_run(self, state):
        if not hasattr(self, 'retriever'):
            raise ValueError("langgraph_prepare needed before langgraph_run")
        docs = await self.retriever.ainvoke(state['question'])
        return {'documents': docs}

