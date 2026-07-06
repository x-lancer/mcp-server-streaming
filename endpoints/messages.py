import json
from typing import Mapping
import logging
from werkzeug import Request, Response
from dify_plugin import Endpoint
from dify_plugin.config.logger_format import plugin_logger_handler

from .auth import validate_bearer_token
from .shared import build_chat_inputs, collect_streaming_result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


class MessageEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
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
            return self._handle_tool_call(data, settings, app_id, tool, session_id)

        else:
            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "error": {"code": -32001, "message": "unsupported method"},
            }

        self.session.storage.set(session_id, json.dumps(response).encode())
        return Response("", status=202, content_type="application/json")

    def _handle_tool_call(self, data, settings, app_id, tool, session_id):
        """Handle tools/call for SSE transport.

        When response-mode is streaming, delegates the Dify call to the SSE
        generator (which runs in sse.py). The generator will make the call
        inline and yield progressive chunks.

        Returns:
            A dict response (to be stored in session storage), OR a Response
            object directly for streaming mode.
        """
        request_id = data.get("id")
        tool_name = data.get("params", {}).get("name")
        arguments = data.get("params", {}).get("arguments", {})

        try:
            if tool_name != tool.get("name"):
                raise ValueError(f"Unknown tool: {tool_name}")

            if settings.get("app-type") == "chat":
                query, inputs = build_chat_inputs(arguments)

                if settings.get("response-mode") == "streaming":
                    # Delegate to SSE generator for progressive streaming
                    call_params = {
                        "app_id": app_id,
                        "query": query,
                        "inputs": inputs,
                        "request_id": request_id,
                    }
                    self.session.storage.set(
                        f"{session_id}_stream_call",
                        json.dumps(call_params).encode()
                    )
                    return Response("", status=202, content_type="application/json")

                # Blocking mode: collect full answer and store
                result = self.session.app.chat.invoke(
                    app_id=app_id, query=query, inputs=inputs,
                    response_mode="streaming",
                )
                answer = collect_streaming_result(result)
                final_result = {"type": "text", "text": answer}
            else:
                result = self.session.app.workflow.invoke(
                    app_id=app_id, inputs=arguments, response_mode="blocking"
                )
                r = [
                    v
                    for v in result.get("data").get("outputs", {}).values()
                    if isinstance(v, str)
                ]
                final_result = {"type": "text", "text": "\n".join(r)}

            logger.info(f"Invoke dify app result answered")
            self.session.storage.set(
                session_id,
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [final_result], "isError": False},
                }).encode()
            )
            return Response("", status=202, content_type="application/json")

        except Exception as e:
            logger.error(f"MessageEndpoint tool call error: {e}")
            self.session.storage.set(
                session_id,
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": str(e)},
                }).encode()
            )
            return Response("", status=202, content_type="application/json")
