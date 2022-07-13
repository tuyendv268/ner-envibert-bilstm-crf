from infer import infer_entity
from fastapi import FastAPI

app = FastAPI()

@app.get("/infer")
def infer(input: str):
    output = infer_entity(input)

    return {
        "input":input,
        "output": output,
        }