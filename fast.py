from fastapi import FastAPI
from pydantic import BaseModel
from generative import generate
import uvicorn

class User_input(BaseModel):
    prompt: str

app = FastAPI()

# academix = GenerativePromptChatAssistant()
@app.post("/generate")
def out(input:User_input):
    return generate(prompt=input.prompt)


# if __name__ == '__main__':
    # uvicorn.run(app, host='0.0.0.0', port=8000)

    