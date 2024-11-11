import json

from langflow.custom import Component
from langflow.io import Output, StateInput, PromptInput
from langflow.base.experts_base.generic_pdf import PDFLoader
from langflow.schema.data import State

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_ollama import ChatOllama


# TODO this must be from some configuration worker multi config to setup multiple
# llm, and this can be utilized the same way in different experts/workers/agents.



class AnswerGrader(Component):
    display_name = "Answer Grader"
    description = "Expert in grading answer for asked question based on rule set."
    icon = "notepad-text"
    name = "answer-grader-llm"


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _instruct_llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0.2, format='json')
        _instruct_llm = _instruct_llm.with_config(tags=["aiinstruct"])
        self.llm_instruct = _instruct_llm


    inputs = [
        StateInput(
            name='state',
            display_name='from',
            info="Link to state or logic box",
            is_list=True
            
        ),
        PromptInput(name="grading_instruction", display_name="Grading Instruction"),
        PromptInput(name="prompt_template", display_name="Grading Template"),
        ]
    outputs = [
        Output(display_name="grader_response", name="grader_response", method="dry_run"),
    ]



    async def grader(self, state):
        question = state["question"]
        documents = state["documents"]
        answer = state["answer"]

        # Score each doc
        documents_text = "\n\n".join(doc.page_content for doc in documents)
        prompt = self.prompt_template.format(
            documents=documents_text, question=question, answer=answer
        )
        result = self.llm_instruct.invoke(
            [SystemMessage(content=self.grading_instruction)]
            + [HumanMessage(content=prompt)]
        )
        # print("Flag grade documents\n", "documents: ", d.page_content, "\nResult:", result)
        answer_score = json.loads(result.content).get("answer_score", 0)
        answer_grader_explanation = json.loads(result.content).get("explanation", "")
        return {"answer_score": answer_score, "answer_grader_explanation": answer_grader_explanation}
    
    async def dry_run(self) -> State:
        # It's expected previous state will come from one source at a time
        # so picking 0, it's list because to allow multiple from for one state node.
        previous_state = [x for x in self.state if x][0].data
        result = await self.grader(previous_state)
        response = {**previous_state, **result}
        return State(data=response)
    
    async def langgraph_prepare(self):
        pass

    async def langgraph_run(self, state):
        results = await self.grader(state)
        return results

