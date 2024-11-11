import ast
import os
import pandas as pd

from loguru import logger

from langflow.utils.constants import DIRECT_TYPES
from langflow.services.deps import get_storage_service
from langflow.utils.util import sync_to_async, unescape_string

class Node():

    def __init__(self, data) -> None:
        self.data = data

    def build_params(self):


        template_dict = {key: value for key, value in self.data["node"]["template"].items() if isinstance(value, dict)}
        params = {}

        # for edge in self.edges:
        #     if not hasattr(edge, "target_param"):
        #         continue
        #     params = self._set_params_from_normal_edge(params, edge, template_dict)

        load_from_db_fields = []
        
        for field_name, field in template_dict.items():
            if field_name in params:
                continue
            # Skip _type and any value that has show == False and is not code
            # If we don't want to show code but we want to use it
            if 'type' not in field:
                field['type'] = field['field_type']
            if field_name == "_type" or (not field.get("show") and field_name != "code"):
                continue
            # If the type is not transformable to a python base class
            # then we need to get the edge that connects to this node
            if field.get("type") == "file":
                # Load the type in value.get('fileTypes') using
                # what is inside value.get('content')
                # value.get('value') is the file name
                if file_path := field.get("file_path"):
                    storage_service = get_storage_service()
                    try:
                        flow_id, file_name = os.path.split(file_path)
                        full_path = storage_service.build_full_path(flow_id, file_name)
                    except ValueError as e:
                        if "too many values to unpack" in str(e):
                            full_path = file_path
                        else:
                            raise e
                    params[field_name] = full_path
                elif field.get("required"):
                    field_display_name = field.get("display_name")
                    logger.warning(
                        f"File path not found for {field_display_name} in component {self.display_name}. "
                        "Setting to None."
                    )
                    params[field_name] = None
                else:
                    if field["list"]:
                        params[field_name] = []
                    else:
                        params[field_name] = None

            elif field.get("type") in DIRECT_TYPES and params.get(field_name) is None:
                val = field.get("value")
                if field.get("type") == "code":
                    try:
                        params[field_name] = ast.literal_eval(val) if val else None
                    except Exception:
                        params[field_name] = val
                elif field.get("type") in ["dict", "NestedDict"]:
                    # When dict comes from the frontend it comes as a
                    # list of dicts, so we need to convert it to a dict
                    # before passing it to the build method
                    if isinstance(val, list):
                        params[field_name] = {k: v for item in field.get("value", []) for k, v in item.items()}
                    elif isinstance(val, dict):
                        params[field_name] = val
                elif field.get("type") == "int" and val is not None:
                    try:
                        params[field_name] = int(val)
                    except ValueError:
                        params[field_name] = val
                elif field.get("type") == "float" and val is not None:
                    try:
                        params[field_name] = float(val)
                    except ValueError:
                        params[field_name] = val
                        params[field_name] = val
                elif field.get("type") == "str" and val is not None:
                    # val may contain escaped \n, \t, etc.
                    # so we need to unescape it
                    if isinstance(val, list):
                        params[field_name] = [unescape_string(v) for v in val]
                    elif isinstance(val, str):
                        params[field_name] = unescape_string(val)
                    elif isinstance(val, Data):
                        params[field_name] = unescape_string(val.get_text())
                elif field.get("type") == "bool" and val is not None:
                    if isinstance(val, bool):
                        params[field_name] = val
                    elif isinstance(val, str):
                        params[field_name] = val != ""
                elif field.get("type") == "table" and val is not None:
                    # check if the value is a list of dicts
                    # if it is, create a pandas dataframe from it
                    if isinstance(val, list) and all(isinstance(item, dict) for item in val):
                        params[field_name] = pd.DataFrame(val)
                    else:
                        raise ValueError(f"Invalid value type {type(val)} for field {field_name}")
                elif val is not None and val != "":
                    params[field_name] = val

                elif val is not None and val != "":
                    params[field_name] = val
                if field.get("load_from_db"):
                    load_from_db_fields.append(field_name)

            if not field.get("required") and params.get(field_name) is None:
                if field.get("default"):
                    params[field_name] = field.get("default")
                else:
                    params.pop(field_name, None)
        # Add _type to params
        self.params = params
        self.load_from_db_fields = load_from_db_fields
        self._raw_params = params.copy()
        return params