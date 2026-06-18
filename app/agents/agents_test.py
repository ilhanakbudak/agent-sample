from langchain_core.messages import HumanMessage, ToolMessage
from app.logging_config import setup_logging
from app.llm import get_llm
from app.agents.tools import TOOLS, TOOLS_BY_NAME

setup_logging()
llm = get_llm()
llm_with_tools = llm.bind_tools(TOOLS)          # tell the model what tools exist

messages = [HumanMessage("What is 4825 * 391?")]

ai = llm_with_tools.invoke(messages)            # STEP 1: model decides
messages.append(ai)
print("Tool calls the model requested:", ai.tool_calls)

for call in ai.tool_calls:                       # STEP 2: we execute
    result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final = llm_with_tools.invoke(messages)          # STEP 3: model answers with the result
print("FINAL:", final.content)


#plain = get_llm()
#print(plain.invoke("What is 48251 * 39174? Answer with just the number, no tools.").content)