from ..components.experts import (
    GenerateFromDocuments, DocumentGrader, PDFGeneric, WebGeneric,
    AnswerGrader
)
from ..components.logicbox import FixedLogicBox
from ..components.inputs import TextInputComponent, GraphStateInput
from ..components.outputs import Dispatch
# import importlib

# GenerateFromDocuments = importlib.import_module('langflow.components.experts.generate_from_documents')

components = [
    GenerateFromDocuments,
    DocumentGrader,
    PDFGeneric,
    FixedLogicBox,
    TextInputComponent,
    GraphStateInput,
    WebGeneric,
    AnswerGrader,
    Dispatch
]

mapper = {
    component.name: component for component in components
}

ClsMapper = mapper