from langflow.base.io.text import TextComponent
from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.schema.message import Message
from langflow.schema import Data


class DatastoreInteract(Component):
    display_name = "Datastore Interact"
    description = "Interact with Datastore"
    icon = "datastore"
    name = "datastore-interact"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            info="Text to be passed as input.",
        ),
    ]
    outputs = [
        Output(display_name="Result", name="result", method="build_output"),
    ]

    def build_output(self) -> Data:
        data = Data(value={"a": 1, "b": [12, 3]})
        self.status = data
        return data