from pathlib import Path

from langflow.base.io.text import TextComponent
from langflow.base.data.utils import TEXT_FILE_TYPES, parse_text_file_to_data
from langflow.custom import Component
from langflow.io import Output, FileInput, BoolInput, StateInput, IntInput
from langflow.schema.data import State
from langflow.base.experts_base.generic_pdf import PDFLoader



class PDFGeneric(Component):
    display_name = "Generic PDF Expert"
    description = "Expert in pdfs which analyze and provide document information"
    icon = "file-text"
    name = "generic-pdf-expert"
    SUPPORTED_FILE_TYPES = ["pdf"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    inputs = [
        FileInput(
            name="paths",
            display_name="Paths",
            file_types=SUPPORTED_FILE_TYPES,
            list=True,
            required=True,
            info=f"Supported file types: {', '.join(SUPPORTED_FILE_TYPES)}",
        ),
        IntInput(
            name='no_of_documents',
            display_name='Number of Documents',
        ),
        StateInput(
            name='state',
            display_name='From',
            required=True,
            is_list=True
        ),
        BoolInput(
            name="silent_errors",
            display_name="Silent Errors",
            advanced=True,
            info="If true, errors will not raise an exception.",
        ),
        ]

    outputs = [
        Output(display_name="Data", name="data", method="dry_run"),
    ]



    async def pdf_retriever(self):
        paths = [self.paths]
        if not paths:
            raise ValueError("Please, upload a file to use this component.")
        resolved_paths = []
        for path in paths:
            resolved_paths.append(self.resolve_path(path))

        extensions = [Path(resolved_paths).suffix[1:].lower() for resolved_paths in resolved_paths]

        unsupported_extensions = set(extensions) - set(self.SUPPORTED_FILE_TYPES)

        if unsupported_extensions:
            raise ValueError(f"Unsupported file type: {', '.join(unsupported_extensions)}")

        retriever = await PDFLoader(resolved_paths, ollama_embedder_model='mxbai-embed-large').retreiver(num_of_docs=4)
        return retriever

    
    async def dry_run(self) -> State:
        previous_state = [x for x in self.state if x][0].data
        state = {
            'question': previous_state['question']
        }
        retriever = await self.pdf_retriever(state)
        docs = await retriever.ainvoke(state['question'])
        response = {**previous_state, **{'documents': docs}}
        return State(data=response, text_key='documents')
    
    async def langgraph_prepare(self):
        self.retriever = await self.pdf_retriever()
        print("Inside langgraph_prepare")


    async def langgraph_run(self, state):
        if not hasattr(self, 'retriever'):
            raise ValueError("langgraph_prepare needed before langgraph_run")
        docs = await self.retriever.ainvoke(state['question'])
        return {'documents': docs}

