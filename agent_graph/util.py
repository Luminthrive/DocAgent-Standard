import json
from typing import Optional, Any, Mapping

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from config import config

def get_llm(temperature:float=0.7,max_tokens:Optional[int]=None):
    kwargs=dict()
    if max_tokens:
        kwargs["max_tokens"]=max_tokens
    return ChatOpenAI(
        model=config.llm_model,
        temperature=temperature,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        max_retries=config.llm_retries,
        **kwargs
    )

def get_last_tool_result(messages:list,tool_name:str)->Any:
    for msg in reversed(messages):
        if isinstance(msg,ToolMessage) and msg.name==tool_name:
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError,TypeError):
                return msg.content
    return None

def get_last_user_message(messages:list)->str:
    for msg in reversed(messages):
        if isinstance(msg,HumanMessage):
            return str(msg.content)
    return ""

def estimate_tokens(text:str)->int:
    if not text:
        return 0
    cn_chars=sum(1 for c in text if "一"<=c<="鿿")
    other_chars=len(text)-cn_chars
    return int(cn_chars/1.5+other_chars/4)

def truncate_tool_result(tool_name:str,result:Any)->str:
    if isinstance(result,(dict,list)):
        text=json.dumps(result,ensure_ascii=False,default=str)
    else:
        text=str(result)
    limit=2000 if tool_name=="rag_search" else 500
    if len(text)>limit:
        return text[:limit] +f"\n...(截断，共{len(text)}字符)"
    return text

def build_context(state:Mapping)->str:
    token_budget=config.context_token_budget
    messages=state.get("messages",[])
    parts=[]
    used_tokens=0
    query=state.get("current_query","")
    retry_count=state.get("retry_count",0)

    if query:
        task_context=f"用户问题:{query}"
        if retry_count>0:
            task_context+=f"\n（第{retry_count}次重试，已重写查询）"
        parts.append(f"## 当前任务\n{task_context}")
        used_tokens+=estimate_tokens(task_context)
    rag_result=get_last_tool_result(messages,"rag_search")
    if rag_result:
        tool_text=truncate_tool_result("rag_search",rag_result)
        parts.append(f"## 工具返回结果\n{tool_text}")
        used_tokens+=estimate_tokens(tool_text)

    remaining=token_budget-used_tokens
    if remaining>0:
        recent_messages=messages[-config.recent_message_count:]
        lines=[]
        used=0

        for msg in reversed(recent_messages):
            if isinstance(msg,HumanMessage):
                role="用户"
            elif isinstance(msg,AIMessage):
                role="助手"
            else:
                continue
            content=str(msg.content or "")
            if not content:
                continue
            content=content[:config.history_item_max_chars]
            tokens=estimate_tokens(content)

            if used+tokens>remaining:
                break
            lines.append(f"{role}:{content}")
            used+=tokens
        lines.reverse()
        if lines:
            parts.append("##最近对话\n"+"\n".join(lines))
    return "\n\n".join(parts)





