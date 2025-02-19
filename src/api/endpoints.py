import io

from fastapi import UploadFile, File, Form
from PIL import Image
from model.description_pipeline import img_description_pipeline


async def img_description(prompt: str = Form(...), image: UploadFile = File(...)):
    img_content = await image.read()

    image = Image.open(io.BytesIO(img_content))

    description = img_description_pipeline(prompt, image)  # type: ignore

    return {'Response': description}
