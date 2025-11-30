# services/storage/memory_store.py
import json
import re
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

STORAGE_DIR = Path("storage")
MEMORY_FILE = STORAGE_DIR / "memory.json"
SHORT_MEM_LIMIT = 10  # last N messages to keep per session

_fact_patterns = [
    re.compile(r"\bmy name is ([A-Za-z0-9 _\-]+)", re.IGNORECASE),
    re.compile(r"\bi am ([A-Za-z0-9 _\-]+)", re.IGNORECASE),
    re.compile(r"\bi'm ([A-Za-z0-9 _\-]+)", re.IGNORECASE),
    re.compile(r"\bi live in ([A-Za-z0-9 _,\-]+)", re.IGNORECASE),
    re.compile(r"\bi from ([A-Za-z0-9 _,\-]+)", re.IGNORECASE),
    re.compile(r"\bi prefer ([A-Za-z0-9 _,\-]+)", re.IGNORECASE),
    re.compile(r"\bi want to ([A-Za-z0-9 _,\-]+)", re.IGNORECASE),
]

class MemoryStore:
    """
    Simple hybrid memory:
    - sessions: { session_id: [last messages] }
    - users: { user_id: { "facts": {k:v}, "created_at": ts } }
    Persisted to storage/memory.json
    """
    def __init__(self, path: Path = MEMORY_FILE):
        self.path = Path(path)
        self.lock = Lock()
        self.data = {"sessions": {}, "users": {}}
        self._ensure_storage()
        self.load()

    def _ensure_storage(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self.data, f)

    def load(self):
        try:
            with self.path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data = loaded
            else:
                self.data = {"sessions": {}, "users": {}}
        except Exception:
            self.data = {"sessions": {}, "users": {}}

    def save(self):
        with self.lock:
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            Path(tmp).replace(self.path)

    def add_message(self, session_id: str, user_id: str, text: str):
        # short-term per-session
        sess = self.data.setdefault("sessions", {})
        msgs = sess.setdefault(session_id, [])
        msgs.append({"user": user_id, "text": text})
        # trim
        if len(msgs) > SHORT_MEM_LIMIT:
            msgs[:] = msgs[-SHORT_MEM_LIMIT:]
        # extract facts and persist to user-level memory
        if user_id:
            facts = self.data.setdefault("users", {}).setdefault(user_id, {}).setdefault("facts", {})
            extracted = self._extract_facts(text)
            for k, v in extracted.items():
                # simple overwrite: latest wins
                facts[k] = v
        # persist
        self.save()

    def get_session_messages(self, session_id: str) -> List[Dict]:
        return list(self.data.get("sessions", {}).get(session_id, []))

    def get_user_facts(self, user_id: str) -> Dict[str, str]:
        return dict(self.data.get("users", {}).get(user_id, {}).get("facts", {}))

    def get_combined_memory(self, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Returns a compact string summarizing session messages (last few)
        and salient user facts.
        """
        parts = []
        msgs = self.get_session_messages(session_id)
        if msgs:
            last_texts = [m["text"] for m in msgs[-3:]]  # include last 3 messages
            parts.append("Recent: " + " || ".join(last_texts))
        if user_id:
            facts = self.get_user_facts(user_id)
            if facts:
                fstr = ", ".join(f"{k}={v}" for k, v in facts.items())
                parts.append("Known: " + fstr)
        return " | ".join(parts) if parts else ""

    def clear_session(self, session_id: str):
        if "sessions" in self.data and session_id in self.data["sessions"]:
            del self.data["sessions"][session_id]
            self.save()

    def _extract_facts(self, text: str) -> Dict[str, str]:
        found = {}
        if not text:
            return found
        for patt in _fact_patterns:
            m = patt.search(text)
            if m:
                key = patt.pattern.split("\\b")[0] if "\\b" in patt.pattern else patt.pattern
                # make simpler key name
                group = m.group(1).strip()
                # heuristics for key
                low = patt.pattern.lower()
                if "name" in low or "i am" in low:
                    found["name"] = group
                elif "live in" in low or "from" in low:
                    found["location"] = group
                elif "prefer" in low:
                    found["preference"] = group
                elif "want to" in low:
                    found["goal"] = group
                else:
                    k = f"fact_{len(found)+1}"
                    found[k] = group
        return found
