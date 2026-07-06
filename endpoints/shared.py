import json
import logging

logger = logging.getLogger(__name__)


def build_chat_inputs(arguments: dict) -> tuple:
    """Build query and inputs for chat.invoke from MCP tool arguments."""
    query = (
        arguments.get("query")
        or arguments.get("userinput.query")
        or arguments.get("input")
        or ""
    )
    inputs = dict(arguments)
    if "input" not in inputs:
        inputs["input"] = query
    return query, inputs


def collect_streaming_result(stream_result):
    """Collect Dify streaming events into a single answer string."""
    answer = ""
    for event in stream_result:
        if isinstance(event, dict):
            event_type = event.get("event", "")
            if event_type == "message":
                answer = event.get("answer", answer)
            elif event_type == "message_end":
                answer = event.get("answer", answer)
            elif event_type == "agent_message":
                answer += event.get("answer", "")
            elif event_type == "error":
                raise Exception(event.get("message", "Stream error"))
    return answer


def iter_stream_chunks(stream_result):
    """Process Dify streaming events, yielding (accumulated_answer, is_final).

    Used by the SSE transport generator to yield progressive JSON-RPC
    responses as the Dify stream produces events.
    """
    answer = ""
    for event in stream_result:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event", "")

        if event_type == "agent_message":
            answer += event.get("answer", "")
            yield answer, False
        elif event_type == "message":
            answer = event.get("answer", answer)
            yield answer, False
        elif event_type == "message_end":
            answer = event.get("answer", answer)
            yield answer, True
            return
        elif event_type == "error":
            raise Exception(event.get("message", "Stream error"))

    # Safety net: yield whatever we have as final
    yield answer, True
