import uuid
import time
import json
import logging

from typing import Mapping
from werkzeug import Request, Response
from dify_plugin import Endpoint
from dify_plugin.config.logger_format import plugin_logger_handler

from .auth import validate_bearer_token
from .shared import iter_stream_chunks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


def create_sse_message(event, data):
    return f"event: {event}\ndata: {json.dumps(data) if isinstance(data, (dict, list)) else data}\n\n"


class SSEEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        logger.info(f"SSEEndpoint request headers: {r.headers}")

        auth_error = validate_bearer_token(r, settings)
        if auth_error:
            return auth_error

        session_id = str(uuid.uuid4()).replace("-", "")

        def generate():
            endpoint = f"messages/?session_id={session_id}"
            yield create_sse_message("endpoint", endpoint)

            while True:
                # Check for progressive streaming call
                call_key = f"{session_id}_stream_call"
                if self.session.storage.exist(call_key):
                    try:
                        call_params = json.loads(
                            self.session.storage.get(call_key).decode()
                        )
                        self.session.storage.delete(call_key)

                        app_id = call_params["app_id"]
                        query = call_params["query"]
                        inputs = call_params["inputs"]
                        request_id = call_params["request_id"]

                        result = self.session.app.chat.invoke(
                            app_id=app_id,
                            query=query,
                            inputs=inputs,
                            response_mode="streaming",
                        )

                        for answer, is_final in iter_stream_chunks(result):
                            yield create_sse_message(
                                "message",
                                json.dumps({
                                    "jsonrpc": "2.0",
                                    "id": request_id,
                                    "result": {
                                        "content": [{"type": "text", "text": answer}],
                                        "isError": False,
                                    },
                                })
                            )
                    except Exception as e:
                        logger.error(f"SSE streaming call error: {e}")
                        yield create_sse_message(
                            "message",
                            json.dumps({
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {"code": -32000, "message": str(e)},
                            })
                        )

                    continue  # Back to polling loop

                # Regular (non-streaming) message delivery
                if self.session.storage.exist(session_id):
                    message = self.session.storage.get(session_id)
                    message = message.decode()
                    self.session.storage.delete(session_id)
                    yield create_sse_message("message", message)

                time.sleep(0.5)

        return Response(generate(), status=200, content_type="text/event-stream")
