from pydantic import BaseModel, Field


class ObservationResponse(BaseModel):
    joints_deg: list[float] = Field(min_length=6, max_length=6)
    image_jpeg: str
    timestamp_ns: int


class ActionRequest(BaseModel):
    joints_deg: list[float] = Field(min_length=6, max_length=6)
    duration_ms: int = Field(ge=20, le=10_000)


class ActionResponse(BaseModel):
    accepted: bool
    command_id: str
    backend: str
