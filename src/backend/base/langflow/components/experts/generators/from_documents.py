from langflow.custom import Component
from langflow.io import Output, StateInput, PromptInput
from langflow.base.experts_base.generic_pdf import PDFLoader
from langflow.schema.data import State

from langchain_core.messages import HumanMessage

# from langchain_ollama import ChatOllama


# TODO this must be from some configuration worker multi config to setup multiple
# llm, and this can be utilized the same way in different experts/workers/agents.


# model_2 = 'llama3.2'
# llm = ChatOllama(model=model_2, temperature=0.3)
# llm = llm.with_config(tags=["airesponse"])


class GenerateFromDocuments2(Component):
    display_name = "LLM Gen from Documents 2"
    description = "Expert in generate resonse from given documents"
    icon = "lightbulb"
    name = "generate-from-document-2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    inputs = [
        StateInput(
            name='state',
            display_name='from',
            info="Link to state or logic box",
            is_list=True
            
        ),
        PromptInput(name="prompt_template", display_name="Generate Prompt Template"),
        ]
    outputs = [
        # Output(display_name="singluar_value", name="Singular Value", method="build_singular_output"),
        # Output(display_name="data_value", name="JSON Value", method="build_json_output"),
        Output(display_name="llm_gen_response", name="llm_gen_response", method="dry_run"),
    ]



    async def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)
        print("FLag loop_step", retry_count)

        # RAG generation
        docs_txt = "\n\n".join(doc.page_content for doc in documents)
        prompt = self.prompt_template.format(documents_text=docs_txt, question=question)

        # bind here to stream response from tags.
        # generation = await llm.ainvoke([HumanMessage(content=prompt)])

        generation = {'generation': "Hi", 'retry_count': 1}
        return generation

        return {"generation": generation, 'retry_count': retry_count + 1}
    
    async def dry_run(self) -> State:
        # It's expected previous state will come from one source at a time
        # so picking 0, it's list because to allow multiple from for one state node.
        return State(data={'a': 1}, text_key='a')
        previous_state = self.state[0].data
        state = {
            'question': previous_state['question'],
            'documents': previous_state['documents']
        }
        response = await self.generate(state)
        return State(data=response, text_key='generation')

    async def langgraph_run(self, state):
        results = await self.generate(state)
        return results

