from typing import Any

from pydantic import AliasChoices, BaseModel
from pydantic import Field as PydanticField


class HordeModel(BaseModel):
    # model_config = ConfigDict(extra="forbid")

    def model_dump(self, *args, by_alias: bool = True, **kwargs) -> dict[str, Any]:
        return super().model_dump(*args, by_alias=by_alias, **kwargs)

    def model_dump_json(self, *args, by_alias: bool = True, **kwargs) -> str:
        return super().model_dump_json(*args, by_alias=by_alias, **kwargs)


class HordeRequest(HordeModel):
    pass


class HordeResponse(HordeModel):
    """An API response. Child objects should not use this type."""


class HordeSuccess(HordeResponse):
    """A successful API response."""


class HordeRequestError(RuntimeError):
    """An error response from the AI Horde API."""

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


# noinspection PyPep8Naming
def RenamedField(  # noqa: N802
    *args: Any,
    renamed_to: str,
    original_name: str,
    **kwargs: Any,
) -> Any:
    """A renamed field that needs to be serialised and validated as its original name."""
    return PydanticField(
        *args,
        **kwargs,
        alias=original_name,
        serialization_alias=original_name,
        validation_alias=AliasChoices(original_name, renamed_to),
    )
