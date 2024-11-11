import asyncio
import json
import time
import traceback
import typing
import uuid

from typing import TYPE_CHECKING, Annotated
from functools import reduce

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import ContentStream
from starlette.types import Receive

from sqlmodel import Session, and_, col, select

from langflow.api.utils import (
    build_and_cache_graph_from_data,
    build_graph_from_data,
    build_graph_from_db,
    build_graph_from_db_no_cache,
    format_elapsed_time,
    format_exception_message,
    get_top_level_vertices,
    parse_exception,
)
from langflow.api.v1.schemas import (
    FlowDataRequest,
    InputValueRequest,
    ResultDataResponse,
    StreamData,
    VertexBuildResponse,
    VerticesOrderResponse,
)
from langflow.events.event_manager import EventManager, create_default_event_manager
from langflow.exceptions.component import ComponentBuildException
from langflow.graph.graph.base import Graph
from langflow.graph.utils import log_vertex_build
from langflow.schema.schema import OutputValue
from langflow.interface import initialize
from langflow.services.auth.utils import get_current_active_user
from langflow.services.database.models.flow import Flow
from langflow.services.chat.service import ChatService
from langflow.services.deps import get_chat_service, get_session, get_session_service, get_telemetry_service
from langflow.services.telemetry.schema import ComponentPayload, PlaygroundPayload
from langflow.services.telemetry.service import TelemetryService

from langflow.api.v2.utils.node import Node as LangflowNode


if TYPE_CHECKING:
    from langflow.graph.vertex.types import InterfaceVertex
    from langflow.services.session.service import SessionService

router = APIRouter(tags=["LanggraphRun"])


from typing_extensions import TypedDict
from typing import List, Annotated, Sequence

# import importlib

# GenerateFromDocuments = importlib.import_module('langflow.components.experts.generate_from_documents.GenerateFromDocuments')

# components = [
#     GenerateFromDocuments,
#     # DocumentGrader,
#     # PDFGeneric,
#     # FixedLogicBox
# ]

# mapper = {
#     component.name: component for component in components
# }
from ...components.langgraph_cross import ClsMapper
from ...inputs.inputs import instantiate_input

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages, MessagesState, AnyMessage


class GraphState(TypedDict):
    # Comman and mandatory
    question: str

    # Internal
    conversation_history: Annotated[Sequence[BaseMessage], add_messages]
    summarize_history: str

    documents: List[str]

    generation: str  # LLM generation

    grader_response: str

    answer: str
    retry_count: int
    answer_score: int # 0 - 100

    # Possible dynamic hold variable with the help typeddict i.e
    # this internal variables + flow defined GraphState variables.
    flow_variables: dict

class LangGraphRunner():

    def __init__(self, langflow_graph) -> None:
        self.langflow_graph = langflow_graph
        self.graph_data = {'nodes': [], 'edges': []}
        self.start_state = None
        self.end_state = None
        self.workflow = StateGraph(self._get_graph_state_cls())
        self.node_state_obj_map = {}
        self.graph = None

    def _get_graph_state_cls(self):
        # Detect graphState
        # nodes = list(filter(lambda x: x['id'].startswith('GraphState'), self.langflow_graph['nodes']))
        # if not nodes:
        #     raise ValueError("GraphState Input is required for flow.")
        # if len(nodes) > 1:
        #     raise ValueError("Invalid flow, found more than 1 GraphState state.")
        # node = nodes[0]
        # _id = node['id']
        # props = node['data']
        # name = props['type']
        # langflow_node_obj = LangflowNode(data=node['data'])
        # params = langflow_node_obj.build_params()
        # cls = ClsMapper[name]

        return GraphState

    async def node_to_state(self, node):
        _id = node['id']
        props = node['data']
        name = props['type']
        try:
            display_name = props['display_name']
        except KeyError:
            print("display_name not found for ", _id)
            display_name = name
        cls = ClsMapper[name]
        if display_name in [x['name'] for x in self.graph_data['nodes']]:
            display_name = _id 
        if name == 'GraphState':
            self.start_state = _id
        
        inputs_meta = {**props['node']['template']}
        inputs_meta.pop('_type')

        inputs_ = {}
        skipped = []
        langflow_node_obj = LangflowNode(data=node['data'])

        for _, v in inputs_meta.items():
            try:
                _type = v.get('_input_type', '')
                if not _type:
                    skipped.append(_)
                    continue
                # if _type in ['FileInput', 'StateInput']:
                if True:
                    if '_input_type' in v: v.pop('_input_type')
                    if 'load_from_db' in v: v.pop('load_from_db')
                input_ = instantiate_input(_type, v)
                # inputs_[_] = input_
                inputs_[_] = {
                    'params': params
                }
            except Exception as e:
                print(e, _)

        params = langflow_node_obj.build_params()
        if str(_id).startswith('FixedLogicBox'):
            code = inputs_meta['code'].get('value', '')
            cls = await initialize.loading.instantiate_class_fixed_logic(code)
        state_obj = cls()
        state_obj.set_attributes(params)
        self.node_state_obj_map[_id] = state_obj
        await state_obj.langgraph_prepare()
        state = {
            'id': _id,
            'name': display_name,
            'cls': cls,
            'state_obj': state_obj
        }
        return state
    
    def edge_to_edge(self, edge):
        source = edge['source']
        target = edge['target']
        print(source, " --> ", target)
        return {
            'source': source,
            'target': target,

        }
    
    def transform_edges(self, edges):
        direct_edges = []
        conditional_edges = []
        edges_dict = {}
        for edge in edges:
            source = edge['source']
            target = edge['target']
            if source in edges_dict:
                edges_dict[source].append(target)
            else:
                edges_dict[source] = [target]

        # Safe check
        for source, targets in edges_dict.items():
            if source.startswith('FixedLogicBox'):
                source_obj = self.node_state_obj_map[source]
                conditional_edges.append({'source': source, 'condition': source_obj.langgraph_condition_eval, 'targets': {v: v for v in targets}})
            else:
                if len(targets) > 1:
                    raise ValueError(
                        f"Node {source} target is multinode which "
                        "should be connected using FixedLogic Node."
                    )
                direct_edges.append({'source': source, 'target': targets[0]})

        return direct_edges, conditional_edges
        
    
    async def build_graph(self):
        states = self.graph_data['nodes']
        for state in states:
            self.workflow.add_node(state['id'], state['state_obj'].langgraph_run)
        edges = self.graph_data['edges']
        direct_edges, conditional_edges = self.transform_edges(edges)
        for edge in direct_edges:
            self.workflow.add_edge(edge['source'], edge['target'])
        for edge in conditional_edges:
            self.workflow.add_conditional_edges(edge['source'], edge['condition'], edge['targets'])
            # self.workflow.add_edge(edge['source'], edge['target'], condition=edge['condition'])
        self.workflow.set_entry_point(self.start_state)
        graph = self.workflow.compile()
        graph.get_graph().draw_png(output_file_path='/Users/furkan/Desktop/LIS/IRWorker/display2.png')
        self.graph = graph


    async def langflow_to_langgraph(self):
        nodes = self.langflow_graph['nodes']
        edges = self.langflow_graph['edges']
        _nodes = []
        for node in nodes:
            transformed_node = await self.node_to_state(node)
            _nodes.append(transformed_node)
        edges = list(map(self.edge_to_edge, edges))
        self.graph_data['nodes'] = _nodes
        self.graph_data['edges'] = edges
        await self.build_graph()

GRAPHS = {}

@router.post("/langgraph/{flow_id}/run")
async def build_flow(
    background_tasks: BackgroundTasks,
    flow_id: uuid.UUID,
    inputs: Annotated[InputValueRequest | None, Body(embed=True)] = None,
    data: Annotated[FlowDataRequest | None, Body(embed=True)] = None,
    files: list[str] | None = None,
    stop_component_id: str | None = None,
    start_component_id: str | None = None,
    log_builds: bool | None = True,
    chat_service: "ChatService" = Depends(get_chat_service),
    current_user=Depends(get_current_active_user),
    telemetry_service: "TelemetryService" = Depends(get_telemetry_service),
    session=Depends(get_session),
):
    """Langgraph run endpoint."""
    # auth_settings = settings_service.auth_settings
    flow_id_str = str(flow_id)
    # graph = build_graph_from_data(flow_id_str, data.model_dump())
    # graph = await build_graph_from_data(flow_id_str, {})
    stmt = select(Flow).where(Flow.id == flow_id)
    # if auth_settings.AUTO_LOGIN:
    #     # If auto login is enable user_id can be current_user.id or None
    #     # so write an OR
    #     stmt = stmt.where(
    #         (Flow.user_id == current_user.id) | (Flow.user_id == None)  # noqa
    #     )  # noqa
    if user_flow := session.exec(stmt).first():
        pass
    else:
        raise HTTPException(status_code=404, detail="Flow not found")
    user_flow = session.exec(stmt).first()
    langgraph_runner = LangGraphRunner(langflow_graph=user_flow.data)
    await langgraph_runner.langflow_to_langgraph()
    global GRAPHS
    GRAPHS = {
        flow_id: langgraph_runner.graph
    }
    return {"status": "ok", "message": "langgraph built successfully"} 

from pydantic import BaseModel
class UserInput(BaseModel):
    query: str
    user_id: str=""

class UserInputNlux(BaseModel):
    input: str
    user_id: str=""

@router.post("/langgraph/{flow_id}/interact")
async def interact(
    flow_id: uuid.UUID,
    user_input: UserInput
):
    """Langgraph interact endpoint."""
    global GRAPHS
    graph = GRAPHS[flow_id]
    inputs = {"question": user_input.query}
    print("Flag query", user_input.query)
    config = {}
    # return {"message": "awaited"}
    async for event in graph.astream_events(inputs, config, version="v2"):
        kind = event["event"]
        tags = event.get("tags", [])
        # filter on the custom tag
        # print(event, end="\n")
        if kind == "on_chat_model_stream" and "airesponse" in event.get("tags", []):
            data = event["data"]
            if data["chunk"].content:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(data["chunk"].content, end="", flush=True)
    return {"message": "success"}

from typing import AsyncGenerator, cast

async def stream_with_errors(generator: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    try:
        async for chunk in generator:
            yield chunk
    # except Exception as e:
    #     body = cast(dict, e.body)
    #     error_msg = body.get("message", "OpenAI API rate limit exceeded")
    #     yield f"event: error_event\ndata: {error_msg}\n\n"
    
    # You can have your own custom exceptions.
    except Exception as e:
        # you may choose to return a generic error or risk it with str(e)
        error_msg = "An error occurred."
        yield error_msg

@router.post("/langgraph/{flow_id}/stream")
async def interact(
    flow_id: uuid.UUID,
    user_input: UserInputNlux
):
    """Langgraph interact endpoint."""
    global GRAPHS
    graph = GRAPHS[flow_id]
    inputs = {"question": user_input.input}
    print("Flag query", user_input.input)
    config = {}
    # return {"message": "awaited"}

    async def stream_responses():
        async for event in graph.astream_events(inputs, config, version="v2"):
            kind = event["event"]
            tags = event.get("tags", [])
            # filter on the custom tag
            # print(event, end="\n")
            if kind == "on_chat_model_stream" and "airesponse" in event.get("tags", []):
                data = event["data"]
                if data["chunk"].content:
                    # Empty content in the context of OpenAI or Anthropic usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(data["chunk"].content, end="", flush=True)
                    yield data['chunk'].content

    responses = stream_responses()
    sr = StreamingResponse(responses, media_type='text/event-stream')
    # sr['Cache-Control'] = 'no-cache'
    # sr['X-Accel-Buffering'] = 'no'
    return sr
    # return {"message": "success"}