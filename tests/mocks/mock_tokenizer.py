"""Pass-through tokenizer step for model-free processor tests."""

from lerobot.processor import EnvTransition, ProcessorStep


class MockTokenizerProcessorStep(ProcessorStep):
    """Mock tokenizer processor step for testing."""

    def __init__(self, *args, **kwargs):
        # Accept any arguments to mimic the real TokenizerProcessorStep interface.
        pass

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(self, features):
        return features
