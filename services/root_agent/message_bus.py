# services/root_agent/message_bus.py
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Msg:
    sender: str
    type: str
    payload: Dict[str, Any]
    reply_to: Optional[str] = None

class MessageBus:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.lock = asyncio.Lock()

    async def register(self, name: str, q: asyncio.Queue):
        async with self.lock:
            self.queues[name] = q

    async def send(self, msg: Msg):
        # deliver to reply_to if present, else to payload.to if present, else broadcast to root
        dest = msg.reply_to or msg.payload.get("to")
        if dest and dest in self.queues:
            await self.queues[dest].put(msg)
            return
        # broadcast to root
        if "root" in self.queues:
            await self.queues["root"].put(msg)
        else:
            print("[bus] no recipient for", msg)
