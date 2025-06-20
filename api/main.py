from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import audit

app = FastAPI()

@app.post("/audit")
async def audit_endpoint(
    image: UploadFile = File(...),
    platform: str = Form(...),
):
    data = await image.read()
    result = audit.analyze(data, platform)
    return JSONResponse(content=result)

