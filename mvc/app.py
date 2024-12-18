from fastapi import FastAPI
from controllers.predict_controller import router as predict_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the predict router
app.include_router(predict_router)

@app.get("/")
async def main():
    return "Hello, I am alive"
