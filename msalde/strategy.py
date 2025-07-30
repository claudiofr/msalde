class AcqusitionStrategy:
    """
    Base class for acquisition strategies.
    """
    def __init__(self, name: str, parameters: dict):
        self._name = name
        self._parameters = parameters

    def select(self, model, data):
        """
        Select the next data point to query based on the model and current data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")