from pydantic import BaseModel

from typing import Any, Dict, List, Optional
class InferenceResponse(BaseModel):
    id: Optional[str]
    prediction: Optional[List[float]]
    is_drift: Optional[int]
response=InferenceResponse()