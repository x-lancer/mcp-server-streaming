import json
from typing import Mapping
import logging
from werkzeug import Request, Response
from dify_plugin import Endpoint
from dify_plugin.config.logger_format import plugin_logger_handler

from .auth import validate_bearer_token

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


def _collect_streaming_result(stream_result):
    """Collect streaming result from chat.invoke(response_mode='streaming')"""
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


def _build_chat_inputs(arguments: dict) -> tuple:
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


class MessageEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        Invokes the endpoint with the given request.
        """
        logger.info(f"MessageEndpoint request headers: {r.headers}")
        logger.info(f"MessageEndpoint request json: {r.json}")

        auth_error = validate_bearer_token(r, settings)
        if auth_error:
            return auth_error
        
        app_id = settings.get("app").get("app_id")
        try:
            tool = json.loads(settings.get("app-input-schema"))
        except json.JSONDecodeError:
            logger.error(f'Invalid app-input-schema: {settings.get("app-input-schema")}')
            raise ValueError("Invalid app-input-schema")

        session_id = r.args.get("session_id")
        data = r.json

        if data.get("method") == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "experimental": {},
                        "prompts": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {"name": "Dify", "version": "1.3.0"},
                },
            }

        elif data.get("method") == "ping":
            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {},
            }

        elif data.get("method") == "notifications/initialized":
            return Response("", status=202, content_type="application/json")

        elif data.get("method") == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {"tools": [tool]},
            }

        elif data.get("method") == "tools/call":
            tool_name = data.get("params", {}).get("name")
            arguments = data.get("params", {}).get("arguments", {})

            try:
                if tool_name == tool.get("name"):
                    if settings.get("app-type") == "chat":
                        query, inputs = _build_chat_inputs(arguments)
                        resp_mode = settings.get("response-mode", "streaming")
                        result = self.session.app.chat.invoke(
                            app_id=app_id,
                            query=query,
                            inputs=inputs,
                            response_mode=resp_mode,
                        )
                        if resp_mode == "streaming":
                            answer = _collect_streaming_result(result)
                        else:
                            answer = result.get("answer", "")
                    else:
                        result = self.session.app.workflow.invoke(
                            app_id=app_id, inputs=arguments, response_mode="blocking"
                        )
                    logger.info(f"Invoke dify app result answered")
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")

                if settings.get("app-type") == "chat":
                    final_result = {"type": "text", "text": answer}
                else:
                    r = [
                        v
                        for v in result.get("data").get("outputs", {}).values()
                        if isinstance(v, str)
                    ]
                    final_result = {"type": "text", "text": "\n".join(r)}

                response = {
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "result": {"content": [final_result], "isError": False},
                }
            except Exception as e:
                logger.error(f"MessageEndpoint tool call error: {e}")
                response = {
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "error": {"code": -32000, "message": str(e)},
                }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "error": {"code": -32001, "message": "unsupported method"},
            }

        self.session.storage.set(session_id, json.dumps(response).encode())
        return Response("", status=202, content_type="application/json")

