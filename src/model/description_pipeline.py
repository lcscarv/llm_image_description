from PIL.Image import Image

from model.pretrained_loader import load_pretrained_models


def img_description_pipeline(prompt: str, image: Image) -> str:
    img_processor, description_model = load_pretrained_models()

    encoding = img_processor(text=prompt, images=image, return_tensors='pt')  # type: ignore

    outputs = description_model(**encoding)

    logits = outputs.logits

    index = logits.argmax(-1).item()

    return description_model.config.id2label[index]
