from optiflowx.core import SearchSpace, ModelWrapper

class CustomModelConfig:
    """Configuration class for a user-defined custom ML model."""

    name = "custom_model"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        # Example: Add your custom hyperparameters here
        s.add("custom_param1", "continuous", [0.001, 1.0], log=True)
        s.add("custom_param2", "discrete", [1, 10])
        s.add("custom_param3", "categorical", ["option1", "option2", "option3"])
        return s

    @staticmethod
    def get_wrapper():
        from optiflowx.models.wrappers.custom_model import CustomModel
        return ModelWrapper(CustomModel)
