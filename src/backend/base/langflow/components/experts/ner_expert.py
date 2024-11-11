import json

from datetime import datetime

from langflow.custom import Component
from langflow.io import Output, StateInput, PromptInput, DictInput
from langflow.base.experts_base.generic_pdf import PDFLoader
from langflow.schema.data import State
from langflow.inputs.inputs import HandleInput

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_ollama import ChatOllama


# TODO this must be from some configuration worker multi config to setup multiple
# llm, and this can be utilized the same way in different experts/workers/agents.



class NERExpert(Component):
    display_name = "NER Expert"
    description = "Expert in fetching ner from given context"
    icon = "dices"
    name = "ner-expert"
    llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0.4, format='json')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0.4, format='json')
        _llm = _llm.with_config(tags=["instruct_response"])
        _llm_generate = ChatOllama(model="llama3.2:latest", temperature=0.3)
        _llm_generate = _llm_generate.with_config(tags=["airesponse"])
        self.llm = _llm
        self.generation_llm = _llm_generate
        self.extraction_system_instruction = "You are expert in information extraction from given question."
        self.interact_system_instruction = "You are expert in followup question generation to collect asked information."



    inputs = [
        StateInput(
            name='state',
            display_name='from',
            info="Link to state or logic box",
            is_list=True
            
        ),
        PromptInput(name="extract_prompt", display_name="Extract Prompt Instruction",
            value=(
                "Extract the following 'entities' given their name and description: \n"
                "'entities' from the asked question 'question'.\n\n"
                "entities:\n{entities}\n\n"
                "question: {question}\n\n"
                "Only return those keys which found in given question.\n"
                "Consider today date is {date}\n"
                "Return the answer in format of JSON."
                )
            ),
        PromptInput(name="interact_prompt", display_name="Interact Prompt Instruction",
        value=(
            "Carefully observe the given 'entities' and form question for collection entities information (where entity name and their description given).\n"
            "Here are given entities: \n"
            "{entities}"
            )
        ),
        DictInput(
            name="ner_elements",
            display_name="NER Elements",
            info="NER entity name and their description",
            is_list=True,
        ),
        # HandleInput(
        #     name="ner_tools",
        #     display_name="Language Model",
        #     input_types=["ToolEnabledLanguageModel"],
        #     required=True,
        # ),
        ]
    outputs = [
        # Output(display_name="singluar_value", name="Singular Value", method="build_singular_output"),
        # Output(display_name="data_value", name="JSON Value", method="build_json_output"),
        Output(display_name="Extarcted NER", name="extract-ner", method="dry_run"),
    ]



    async def extract(self, state):
        question = state["question"]
        ner_elements = state["ner_elements"]
        entities_str = ""
        for k, v in self.ner_elements.items():
            entities_str += f"{k} -> {v}\n"
        if '{date}' in self.extract_prompt:
            prompt = self.extract_prompt.format(entities=entities_str, question=question, date=datetime.now().date().strftime('%d-%b-%Y'))
        else:
            prompt = self.extract_prompt.format(entities=entities_str, question=question)

        # bind here to stream response from tags.
        response = await self.llm.ainvoke([SystemMessage(content=self.extraction_system_instruction)]+[HumanMessage(content=prompt)])
        found_entities = json.loads(response.content)
        not_found_entities = {}
        for k, v in self.ner_elements.items():
            if k not in found_entities:
                not_found_entities[k] = v
                continue
            v = found_entities[k]
            if not v:
                # TODO check here if mandatory and default stuff
                not_found_entities[k] = v
        follow_up_question = ""
        if not_found_entities:
            not_found_entities_str = "\n".join([f"{k} -> {v}" for k, v in not_found_entities.items()])
            try:
                response = await self.generation_llm.ainvoke(
                    [SystemMessage(content=self.interact_system_instruction)
                    ]+[HumanMessage(content=self.interact_prompt.format(entities=not_found_entities_str))])
                print("Flag 2", response.content)
                follow_up_question = response.content
            except KeyError:
                follow_up_question = ""


        return {"entities": found_entities, 'not_found_entities': not_found_entities, 'followup_question': follow_up_question}
    
    async def dry_run(self) -> State:
        # It's expected previous state will come from one source at a time
        # so picking 0, it's list because to allow multiple from for one state node.

        previous_state = [x for x in self.state if x][0].data
        state = {
            'question': previous_state['question'],
            'ner_elements': self.ner_elements
        }
        response = await self.extract(state)
        results = {**previous_state, **response}
        return State(data=results, text_key='generation')

    async def langgraph_prepare(self):
        pass

    async def langgraph_run(self, state):
        results = await self.generate(state)
        return results

