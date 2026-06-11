import os
import yaml

class ModelManager:
    def __init__(self, settings_path):
        self.settings_path = settings_path
        self.settings = self._load_settings()

    def _load_settings(self):
        with open(self.settings_path, 'r') as f:
            return yaml.safe_load(f)

    def get_active_model_path(self):
        """Returns the path to the currently active model."""
        model_path = self.settings.get('models', {}).get('active_model')
        if not model_path:
            raise ValueError("No active model specified in settings.")
        return model_path

    def cleanup_old_models(self):
        """Deletes models specified in the delete_models list in settings."""
        models_to_delete = self.settings.get('models', {}).get('delete_models', [])
        for model_file in models_to_delete:
            if os.path.exists(model_file):
                try:
                    os.remove(model_file)
                    print(f"Deleted old model: {model_file}")
                except Exception as e:
                    print(f"Error deleting {model_file}: {e}")
            else:
                print(f"Model {model_file} not found, skipping deletion.")

    def load_active_model(self, model_class):
        """
        Placeholder function to load the model.
        In the real system, you would instantiate your Quoridor model here 
        using torch.load() on the active_model_path.
        """
        path = self.get_active_model_path()
        print(f"Loading model from {path}...")
        # e.g., return model_class.load_from_checkpoint(path)
        return None
