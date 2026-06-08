from fastapi import APIRouter

from app.api.v1 import experiments
from app.api.v1.analysis import router as analysis_router
from app.api.v1.datasets import router as datasets_router
from app.api.v1.projects import router as projects_router
from app.api.v1.health import router as health_router


api_router = APIRouter()

api_router.include_router(health_router, tags=["health"])
api_router.include_router(projects_router, prefix="/projects", tags=["projects"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
api_router.include_router(experiments.router)