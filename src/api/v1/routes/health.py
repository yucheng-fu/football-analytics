from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", tags=["Health"])
def health(request: Request) -> dict[str, bool]:
    service = getattr(request.app.state, "model_service", None)
    model_loaded = service is not None and service.is_model_available()
    return {"model_loaded": model_loaded}
