import uuid
import json
import os

class DocumentStore:
    def __init__(self, store_path="documents.json"):
        self.store_path = store_path

        # Load or create doc store
        if os.path.exists(self.store_path):
            with open(self.store_path, "r") as f:
                self.documents = json.load(f)
        else:
            self.documents = {}

    def save(self):
        with open(self.store_path, "w") as f:
            json.dump(self.documents, f)

    def add_document(self, text: str, filename: str):
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "text": text
        }
        self.save()
        return doc_id

    def get_document(self, doc_id: str):
        return self.documents.get(doc_id)

    def all_documents(self):
        return list(self.documents.values())
