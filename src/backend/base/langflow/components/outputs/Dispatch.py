from langflow.base.io.text import TextComponent
from langflow.custom import Component
from langflow.io import MultilineInput, Output, StateInput
from langflow.schema.message import Message
from langflow.schema.data import State, Data


class Dispatch(Component):
    display_name = "State Dispatch"
    description = "Final State Output/Dispatch"
    icon = "send-horizontal"
    name = "StateDispatch"

    inputs = [
        StateInput(
            name='state',
            display_name='from',
            info="from state",
            is_list=True
            
        ),
    ]
    outputs = [
        Output(display_name="Data", name="data", method="state_to_dispatch_output"),
    ]

    def state_to_dispatch_output(self) -> Data:
        previous_state = [x for x in self.state if x][0].data
        dispatch_data = Data(**previous_state)
        return dispatch_data
    
    async def langgraph_prepare(self):
        pass
    

    async def langgraph_run(self, state):
        print("Final Dispatch")
        return state

