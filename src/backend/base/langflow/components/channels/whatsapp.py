from langflow.base.io.text import TextComponent
from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.schema.message import Message
from langflow.schema import Data


class Whatsapp(Component):
    display_name = "Whatsapp"
    description = "Integrate Whatsapp channel ok"
    icon = "type"
    name = "WhatsAppInput"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            info="Text to be passed as input.",
        ),
    ]
    outputs = [
        Output(display_name="Text", name="text", method="build_output"),
    ]

    def text_response(self) -> Message:
        message = Message(
            text=self.input_value,
        )
        return message

    def build_output(self) -> Data:
        data = Data(value="acb")
        self.status = data
        return data