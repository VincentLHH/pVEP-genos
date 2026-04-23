from typing import Dict


class MultiEmbeddingManager:
    def __init__(self):
        self.managers = {}

    def add_model(self, model_name, manager):
        self.managers[model_name] = manager

    def get_embeddings(self, seq_dict, methods=["mean"]):
        """
        return:
        {
            "Genos-1.2B": {...},
            "Genos-10B": {...}
        }
        """
        all_results = {}

        for name, manager in self.managers.items():
            print(f"Running model: {name}")
            all_results[name] = manager.get_embeddings(seq_dict, methods)

        return all_results
