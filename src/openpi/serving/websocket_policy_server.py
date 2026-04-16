"""WebSocket server that exposes a trained policy for remote inference."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

import websockets.exceptions
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve

from openpi.policies import policy as _policy
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a :class:`~openpi.policies.policy.Policy` over WebSockets using msgpack + NumPy."""

    def __init__(
        self,
        policy: _policy.Policy,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._packer = msgpack_numpy.Packer()

    async def _handle_client(self, websocket: ServerConnection) -> None:
        await websocket.send(self._packer.pack(self._metadata))
        try:
            async for message in websocket:
                try:
                    obs = msgpack_numpy.unpackb(message)
                    result = self._policy.infer(obs)
                    await websocket.send(self._packer.pack(result))
                except Exception:
                    err = traceback.format_exc()
                    logger.exception("Policy inference failed")
                    await websocket.send(err)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")

    def serve_forever(self) -> None:
        async def runner() -> None:
            async with serve(
                self._handle_client,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
            ) as server:
                logger.info("Policy server listening on ws://%s:%s", self._host, self._port)
                await server.serve_forever()

        asyncio.run(runner())
