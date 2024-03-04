import time
from collections.abc import Hashable
from typing import Any, Literal, TypeVar

from pydantic import AliasChoices, BaseModel
from pydantic import Field as PydanticField


class HordeModel(BaseModel):
    class Config:
        # TODO: THIS EXISTS?????? RenamedField MIGHT BE USELESS!!!!!!!!!!!
        # allow_population_by_field_name = True
        # TODO: Toggle dynamically depending on debug flag?
        # extra = "forbid"
        extra = "allow"

    def model_dump(self, *args, by_alias: bool = True, **kwargs) -> dict[str, Any]:
        return super().model_dump(*args, by_alias=by_alias, **kwargs)

    def model_dump_json(self, *args, by_alias: bool = True, exclude_none: bool = True, **kwargs) -> str:
        return super().model_dump_json(*args, by_alias=by_alias, exclude_none=exclude_none, **kwargs)


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


_H = TypeVar("_H", bound=Hashable)


class SimpleTimedCache:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._cache: dict[Any, dict[Literal["last_accessed", "data"], Any]] = {}

    def _delete_expired(self) -> None:
        now = time.time()
        for key, value in self._cache.items():
            if now - value["last_accessed"] > self.timeout_seconds:
                del self._cache[key]

    def __getattribute__(self, *args, **kwargs) -> Any:
        self._delete_expired()
        return super().__getattribute__(*args, **kwargs)

    def __getitem__(self, key: Hashable) -> Any:
        if key not in self._cache:
            return None
        return self._cache[key]["data"]

    def __setitem__(self, key: Hashable, value: _H) -> _H:
        self._cache[key] = {"last_accessed": time.time(), "data": value}
        return value

    def __delitem__(self, key: Hashable) -> None:
        if key in self._cache:
            del self._cache[key]

    def __contains__(self, item: Hashable) -> bool:
        self._delete_expired()
        return item in self._cache
