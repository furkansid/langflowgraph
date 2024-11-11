from langflow.base.io.text import TextComponent
from langflow.io import MultilineInput, Output, DictInput
from langflow.schema.message import Message
from langflow.schema.data import State


class GraphStateInput(TextComponent):
    display_name = "Graph State"
    description = "Initial Graph State"
    icon = "rows-4"
    name = "GraphState"

    inputs = [
        DictInput(
            name="key_value",
            display_name="Key Value",
            info="key value to be passed as input.",
            is_list=True
        ),
        MultilineInput(
            name="question",
            display_name="Question",
            info="Question to be passed as state.",
            required=True
        ),
    ]
    outputs = [
        Output(display_name="State", name="state", method="inputs_to_state"),
    ]

    def inputs_to_state(self) -> State:
        initial_data = {'question': self.question, **self.key_value}
        initial_state = State(data=initial_data)
        return initial_state
    
    async def langgraph_prepare(self):
        pass
    

    async def langgraph_run(self, state):
        return {'flow_variables': {**self.key_value}}

