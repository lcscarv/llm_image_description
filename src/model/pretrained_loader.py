import os
from typing import Any

from transformers import ViltProcessor, ViltForQuestionAnswering


def load_pretrained_models() -> tuple[tuple[ViltProcessor, dict[str, Any]] | ViltProcessor, ViltForQuestionAnswering]:
    pretrained_path = os.environ['MODEL_PATH']

    img_processor = ViltProcessor.from_pretrained(pretrained_path)
    img_description_model = ViltForQuestionAnswering.from_pretrained(pretrained_path)

    return img_processor, img_description_model
