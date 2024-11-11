import json

from langflow.base.io.text import TextComponent
from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.inputs import DropdownInput, DictInput, MessageTextInput
from langflow.schema.message import Message
from langflow.schema import Data

import redis

redis_io = redis.Redis(host='localhost', port=6379, db=0)


class KeyValueGet(Component):
    display_name = "Key Value Get"
    description = "Key Value Pair flash storage component\n Which allow quick access of variable"
    icon = "datastore"
    name = "key-value-get"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.redis_io = redis.Redis(host='localhost', port=6379, db=0)


    inputs = [
        MultilineInput(
            name="key_name",
            display_name="Key",
            info="Key Name",
        )
        ]
    outputs = [
        # Output(display_name="singluar_value", name="Singular Value", method="build_singular_output"),
        # Output(display_name="data_value", name="JSON Value", method="build_json_output"),
        Output(display_name="data", name="Result", method="build_output"),
    ]

    def _get_value(self):
        value = self.redis_io.get(self.key_name)
        try:
            value = json.loads(value)
        except json.decoder.JSONDecodeError:
            pass
        return value
        
    def build_json_output(self) -> Data:
        return self._get_value()

    def build_singular_output(self) -> Data:
        return self._get_value()
    
    def build_output(self) -> Data:
        value = self._get_value()
        if isinstance(value, dict):
            return Data(data=value)
        else:
            return Data(text=value)