from fastapi import APIRouter


api = APIRouter()


@api.get("/health")
def health_check():
    return "running"
