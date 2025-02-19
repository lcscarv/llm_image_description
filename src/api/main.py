import uvicorn
from fastapi import FastAPI, UploadFile

from app.endpoints import img_description
app = FastAPI()


@app.get("/")
def start():
    return {"LAS DS": "Image Descriptor"}


@app.post("/api/v1/model")
def generate_description(prompt: str, image: UploadFile):
    return img_description(prompt, image)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
