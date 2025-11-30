# services/root_agent/demo_run.py

import logging
import sys
from services.root_agent.root_agent import RootAgent
from services.root_agent.schemas import UserMessage  # <-- matches your schemas.py

logging.basicConfig(level=logging.INFO)

def main():
    root = RootAgent()
    print("EKIS A2A demo (interactive). Ctrl+C to exit.")
    try:
        while True:
            q = input("\nQuery> ").strip()
            if not q:
                continue

            # Build the UserMessage schema object
            msg = UserMessage(
                session_id="demo",
                user_id="demo_user",
                text=q
            )

            # Call the pipeline
            resp = root.handle(msg)

            # Show result
            print("\n--- Agent Response ---")
            print("Intent:", resp.intent)

            res = resp.result
            if isinstance(res, dict) and "answer" in res:
                print("\nAnswer:\n", res["answer"])
            else:
                import json
                print(json.dumps(res, indent=2, ensure_ascii=False))

    except KeyboardInterrupt:
        print("\nBye")
        sys.exit(0)

if __name__ == "__main__":
    main()
