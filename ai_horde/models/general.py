# TODO:
#  Create a custom BaseModel class with an @override'd serialisation that'll specify that aliases should be used
#  Maybe also provide a logger to be used if, for example, the prompt gets trunkated
from pydantic import BaseModel


# TODO: Make use of "validation_alias" for all aliased fields


class HordeModel(BaseModel):
    pass


class HordeRequest(HordeModel):
    pass


class HordeResponse(HordeModel):
    """An API response. Child objects should not use this type"""


class HordeSuccess(HordeResponse):
    """A successful API response"""


class HordeRequestError(RuntimeError):
    """An error response from the AI Horde API"""
    def __init__(
        self,
        message: str | None = None,
        *,
        code: int | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.response_headers = response_headers
        super().__init__(message)
