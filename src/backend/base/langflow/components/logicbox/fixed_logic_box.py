from langflow.custom import Component
from langflow.io import Output, StateInput, HandleInput
from langflow.schema.data import State

class FixedLogicBox(Component):
    display_name = "Fixed Logic Box"
    description = "Fixed Logic Box will work on configured on perfect logic match"
    icon = "logicbox"
    name = "FixedLogicBox"

    inputs = [
        StateInput(
            name='state',
            display_name='from',
            info="Link to state or logic box",
            is_list=True
            
        )
    ]
    outputs = [
        Output(display_name="condition 1", name="condition_1", method="to_condition_1"),
        Output(display_name="condition 2", name="condition_2", method="to_condition_2"),
    ]

    def logic_map(self):
        return {
            # This is not needed for dry-run
            # only for reprensentation and langgraph run
            "resolve_response_1": "to_node_1",
            "resolve_response_2": "to_node_2"
        }
        
        
    def to_condition_1(self) -> State:
        previous_state = self.state[0].data
        condition = False
        # Condition logic here
        if condition:
            return State(data=previous_state)
        else:
            # self.stop("condition_1")
            return None
        
    def to_condition_2(self) -> State:
        previous_state = self.state[0].data
        condition = True
        # Condition logic here
        if condition:
            return State(data=previous_state)
        else:
            # self.stop("condition_1")
            return None
        
    async def langgraph_prepare(self):
        pass

    async def langgraph_run(self, state):
        results = self.logic_map()
        return results
