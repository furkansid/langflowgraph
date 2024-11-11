import json
import redis

from langflow.base.io.text import TextComponent
from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.inputs import DropdownInput, DataInput, MessageTextInput, NestedDictInput
from langflow.schema.message import Message
from langflow.schema import Data


class KeyValueSet(Component):
    display_name = "Key Value Set"
    description = "Key Value Pair flash storage component\n Which allow quick access of variable"
    icon = "datastore"
    name = "key-value-set"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.redis_io = redis.Redis(host='localhost', port=6379, db=0)

    inputs = [
        MultilineInput(
            name="key_name",
            display_name="Key Name",
        ),
        DropdownInput(
        name='value_type',
        display_name="Value Type",
        options=["Singular", "JSON"],
        required=True,
        info="Select store value type"
        ),
        MessageTextInput(
            name="singular_value",
            required=True,
            display_name="Singular Value",
        ),
        NestedDictInput(
            name="data_value",
            display_name="Data Values",
        ),
        ]
    outputs = [
        Output(display_name="status", name="status", method="build_output"),
    ]

    def _store_data_value(self):
        data = json.dumps(self.data_value)
        self.redis_io.set(self.key_name, data)
        return True
    
    def _store_singular_value(self):
        self.redis_io.set(self.key_name, self.singular_value)
        return True

    def build_output(self) -> MessageTextInput:
        if self.value_type == "Singular":
            self._store_singular_value()
        elif self.value_type == "JSON":
            self._store_data_value()
        return "Success"