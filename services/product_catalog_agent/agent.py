class ProductCatalogAgent:
    def __init__(self):
        self.products = [
            {"id": 1, "name": "Sample Product A"},
            {"id": 2, "name": "Sample Product B"}
        ]

    def list_products(self):
        return {"products": self.products}
