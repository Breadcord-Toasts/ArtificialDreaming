import asyncio
import itertools
import math
import time
from collections.abc import AsyncGenerator
from http import HTTPMethod
from json import loads as json_loads
from logging import Logger
from typing import Any, Coroutine, Self, TypeVar
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import aiohttp
from pydantic import BaseModel

from .models.civitai import CivitAIModel, CivitAIModelVersion, ModelType, SortOptions, SearchPeriod
from .models.general import HordeRequest, HordeRequestError, SimpleTimedCache
from .models.horde_meta import ActiveModel, GenerationCheck, GenerationResponse, HordeNews, HordeUser, Team, Worker
from .models.image import (
    FinishedImageGeneration,
    ImageGenerationRequest,
    ImageGenerationStatus,
    InterrogationRequest,
    InterrogationResponse,
    InterrogationStatus,
    InterrogationStatusState,
)
from .models.text import FinishedTextGeneration, TextGenerationRequest, TextGenerationStatus

_T = TypeVar("_T")
AnyModel = TypeVar("AnyModel", bound=BaseModel)


class URL(str):
    def __truediv__(self, other: Any) -> Self:
        """URL / str - add a path to the URL."""
        return self.__class__(f"{self.removesuffix('/')}/{str(other).removeprefix('/')}")

    def __rtruediv__(self, other: Any) -> Self:
        """Str / URL - add a subdomain to the URL."""
        protocol, rest = self.split("://", 1)
        return self.__class__(f"{protocol}://{other!s}.{rest}")

    @property
    def base(self):
        scheme, netloc, *_ = urlparse(self)
        return self.__class__(f"{scheme}://{netloc}")

    @property
    def query_params(self) -> dict[str, str]:
        return dict(parse_qsl(urlparse(self).query))

    def clear_params(self) -> Self:
        return self.__class__(urlparse(self)._replace(query="").geturl())

    def with_params(self, **params: Any) -> Self:
        url = urlparse(self)
        query_params = dict(parse_qsl(url.query))
        query_params.update(params)
        return self.__class__(url._replace(query=urlencode(query_params)).geturl())


HORDE_API_BASE = URL("https://aihorde.net/api/")
CIVITAI_API_DOMAIN = URL("https://civitai.com/api/")

JsonLike = dict[str, "JsonLike"] | list["JsonLike"] | str | int | float | bool | None


class HordeAPI:
    def __init__(self, session: aiohttp.ClientSession, *, logger: Logger):
        self.session = session
        self.logger = logger

    async def get_results(
        self, request: ImageGenerationRequest | TextGenerationRequest, /,
    ) -> list[FinishedImageGeneration] | list[FinishedTextGeneration]:
        if isinstance(request, ImageGenerationRequest):
            generator = self.generate_image
        elif isinstance(request, TextGenerationRequest):
            generator = self.generate_text
        else:
            raise ValueError("Invalid request type")

        *_, generation = [
            generation
            async for generation in generator(request)
        ]
        return generation

    async def generate_image(
        self, generation_settings: ImageGenerationRequest, /,
    ) -> AsyncGenerator[list[FinishedImageGeneration], None]:
        """Simple helper function to both queue an image generation and wait for it to finish."""
        generation = await self.queue_image_generation(generation_settings)
        # There is just about 0 chance that the generation will already be done
        await asyncio.sleep(5)
        async for result in self.wait_for_image_generation(generation):
            yield result

    async def wait_for_image_generation(
        self, queued_generation: GenerationResponse, /,
    ) -> AsyncGenerator[list[FinishedImageGeneration], None]:
        """Yield a list of finished generations for each new image that is generated."""
        start_time = time.time()
        images_done = 0
        while True:
            check = await self.get_image_generation_check(queued_generation.id)
            if check.finished > images_done:
                status = await self.get_image_generation_status(queued_generation.id)
                yield status.generations
            images_done = check.finished
            if check.done or time.time() - start_time > 60 * 10:
                self.logger.debug(f"Image generation finished in {time.time() - start_time:.2f}s")
                return
            await asyncio.sleep(5)

    async def queue_image_generation(self, generation_settings: ImageGenerationRequest, /) -> GenerationResponse:
        queued_generation = GenerationResponse.model_validate(await json_request(
            self.session, HTTPMethod.POST, HORDE_API_BASE / "v2/generate/async",
            data=generation_settings,
        ))
        self.logger.debug(f"Image generation queued: {queued_generation.id}")
        return queued_generation

    async def get_image_generation_status(self, generation_id: str) -> ImageGenerationStatus:
        """Get the status of an image generation. A "full" request will contain generated images."""
        json = await json_request(
            self.session,
            HTTPMethod.GET,
            url=HORDE_API_BASE / "v2/generate/status" / generation_id,
        )
        return ImageGenerationStatus.model_validate(json)

    async def get_image_generation_check(self, generation_id: str) -> GenerationCheck:
        """Get the status of an image generation. A "full" request will contain generated images."""
        json = await json_request(
            self.session,
            HTTPMethod.GET,
            url=HORDE_API_BASE / "v2/generate/check" / generation_id,
        )
        return GenerationCheck.model_validate(json)

    async def cancel_image_generation(self, generation_id: str) -> None:
        await json_request(self.session, HTTPMethod.DELETE, HORDE_API_BASE / "v2/generate/status" / generation_id)

    async def generate_text(
        self, generation_settings: TextGenerationRequest, /,
    ) -> AsyncGenerator[list[FinishedTextGeneration], None]:
        """Simple helper function to both queue a text generation and wait for it to finish."""
        generation = await self.queue_text_generation(generation_settings)
        # There is just about 0 chance that the generation will already be done
        await asyncio.sleep(5)
        async for result in self.wait_for_text_generation(generation):
            yield result

    async def wait_for_text_generation(
        self, queued_generation: GenerationResponse, /,
    ) -> AsyncGenerator[list[FinishedTextGeneration], None]:
        """Yield a list of finished generations for each new text that is generated."""
        start_time = time.time()
        texts_done = 0
        while True:
            check = await self.get_text_generation_status(queued_generation.id)
            if check.finished > texts_done:
                status = await self.get_text_generation_status(queued_generation.id)
                yield status.generations
            texts_done = check.finished
            if check.done or time.time() - start_time > 60 * 10:
                self.logger.debug(f"Text generation finished in {time.time() - start_time:.2f}s")
                return
            await asyncio.sleep(5)

    async def queue_text_generation(self, generation_settings: TextGenerationRequest, /) -> GenerationResponse:
        queued_generation = GenerationResponse.model_validate(await json_request(
            self.session, HTTPMethod.POST, HORDE_API_BASE / "v2/generate/text/async",
            data=generation_settings,
        ))
        self.logger.debug(f"Text generation queued: {queued_generation.id}")
        return queued_generation

    async def get_text_generation_status(self, generation_id: str) -> TextGenerationStatus:
        return TextGenerationStatus.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/generate/text/status" / generation_id,
        ))

    async def cancel_text_generation(self, generation_id: str) -> None:
        await json_request(self.session, HTTPMethod.DELETE, HORDE_API_BASE / "v2/generate/text/status" / generation_id)

    async def interrogate(self, interrogation_settings: InterrogationRequest, /) -> InterrogationStatus:
        """Simple helper function to both queue an interrogation and wait for it to finish."""
        interrogation = await self.queue_interrogation(interrogation_settings)

        while True:
            check = await self.get_interrogation_status(interrogation.id)
            if check.state == InterrogationStatusState.DONE:
                return check
            await asyncio.sleep(min(20, 4 * len(interrogation_settings.forms)))

    async def queue_interrogation(self, interrogation_settings: InterrogationRequest, /) -> InterrogationResponse:
        queued_interrogation = InterrogationResponse.model_validate(await json_request(
            self.session, HTTPMethod.POST, HORDE_API_BASE / "v2/interrogate/async",
            data=interrogation_settings,
        ))
        self.logger.debug(f"Interrogation queued: {queued_interrogation.id}")
        return queued_interrogation

    async def get_interrogation_status(self, interrogation_id: str) -> InterrogationStatus:
        return InterrogationStatus.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/interrogate/status" / interrogation_id,
        ))

    async def cancel_interrogation(self, interrogation_id: str) -> None:
        await json_request(self.session, HTTPMethod.DELETE, HORDE_API_BASE / "v2/interrogate/status" / interrogation_id)

    async def get_models(self) -> list[ActiveModel]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/status/models")
        return [ActiveModel.model_validate(model) for model in json]

    async def get_model(self, model_name: str) -> ActiveModel | None:
        """Get a model by its exact name on the horde (case-sensitive)."""
        json = await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/status/models/{quote(model_name)}",
        )
        if len(json) == 0:
            return None
        return ActiveModel.model_validate(json[0])

    async def get_users(self) -> list[HordeUser]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/users")
        return [HordeUser.model_validate(user) for user in json]

    async def get_user(self, user_id: int) -> HordeUser:
        return HordeUser.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/users/{user_id}",
        ))

    async def get_current_user(self) -> HordeUser:
        return HordeUser.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/find_user",
        ))

    async def get_teams(self) -> list[Team]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/teams")
        return [Team.model_validate(team) for team in json]

    async def get_team(self, team_id: str) -> Team:
        return Team.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/teams/{team_id}",
        ))

    async def get_workers(self) -> list[Worker]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/workers")
        return [Worker.model_validate(worker) for worker in json]

    async def get_worker(self, worker_id: str) -> Worker:
        return Worker.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/workers/{worker_id}",
        ))

    async def get_news(self) -> list[HordeNews]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/status/news")
        return [HordeNews.model_validate(news) for news in json]


class CivitAIAPI:
    def __init__(self, session: aiohttp.ClientSession, *, logger: Logger):
        self.session = session
        self.logger = logger
        self._cache = SimpleTimedCache(60 * 60 * 12)

    async def _fetch_paginated_json_list(self, url: URL, *, pages: int = 1) -> AsyncGenerator[JsonLike, None, None]:
        total_pages: int = -1
        for page in range(1, pages + 1):
            if total_pages != -1 and page > total_pages:
                break
            self.logger.debug(f"Fetching page {page} of a requested {pages}: {url}")
            json: dict = await json_request(self.session, HTTPMethod.GET, url)  # type: ignore[assignment]
            url = json.get("metadata", {}).get("nextPage")
            page_items = json.get("items", [])

            if not page_items or url is None:
                break
            for item in page_items:
                yield item

            # TODO: This might no longer be part of the api?
            metadata_total_pages = json.get("metadata", {}).get("totalPages", total_pages)
            if isinstance(metadata_total_pages, int):
                total_pages = metadata_total_pages

    async def get_models(
        self,
        query: str | None = None,
        *,
        limit: int = 100,
        tag: str | None = None,
        types: list[ModelType] | None = None,
        creator_username: str | None = None,
        period: SearchPeriod | None = None,
        nsfw: bool | None = None,
        sort: SortOptions | None = None,
    ) -> list[CivitAIModel]:
        per_page = 100
        pages = math.ceil(limit / per_page)

        params = dict(
            query=query,
            tags=tag,
            types=[model_type.value for model_type in types] if types else None,
            username=creator_username,
            period=period.value if period else None,
            nsfw=nsfw,
            sort=sort.value if sort else None,
        )
        params = {key: value for key, value in params.items() if value is not None}

        url = CIVITAI_API_DOMAIN / "v1/models"
        url = url.with_params(**params)

        models = []
        async for model in self._fetch_paginated_json_list(url, pages=pages):
            models.append(CivitAIModel.model_validate(model))
        return models

    async def get_model(self, model_id: int | str) -> CivitAIModel | None:
        json = await json_request(self.session, HTTPMethod.GET, CIVITAI_API_DOMAIN / "v1/models" / model_id)
        return CivitAIModel.model_validate(json)

    async def get_model_version(self, version_id: int | str) -> CivitAIModelVersion | None:
        json = await json_request(self.session, HTTPMethod.GET, CIVITAI_API_DOMAIN / "v1/model-versions" / version_id)
        return CivitAIModelVersion.model_validate(json)

    async def get_model_from_version(self, version: CivitAIModelVersion | int | str) -> CivitAIModel | None:
        # Type checker does not know what is going on :3
        if not isinstance(version, CivitAIModelVersion):
            version = await self.get_model_version(version)  # type: ignore
        return await self.get_model(version.model_id)  # type: ignore


async def json_request(
    session: aiohttp.ClientSession,
    method: HTTPMethod,
    url: str,
    *,
    data: dict[str, JsonLike] | BaseModel | None = None,
    headers: dict[str, str] | None = None,
) -> JsonLike:
    """Helper function to make a request with a pydantic model as the data.

    :param session: aiohttp.ClientSession to use
    :param method: HTTP method to use
    :param url: URL to make the request to
    :param data: Data to send, either a pydantic model or a dict that can be serialised to JSON
    :param headers: Additional headers to send
    :return: The JSON response as a dict
    """
    if isinstance(data, HordeRequest):
        response = await session.request(
            method,
            url,
            headers=headers,
            # This is a bit of a hack, but not sure if there's a better way.
            json=json_loads(data.model_dump_json()),
        )
    else:
        response = await session.request(method, url, headers=headers, json=data)
    if not response.ok:
        try:
            json: JsonLike = await response.json()
        except aiohttp.ContentTypeError:
            raise HordeRequestError(
                message=(await response.content.read()).decode(errors="replace") or response.reason,
                code=response.status,
                response_headers=dict(response.headers),
            )
        message: str | None = None
        if isinstance(json, dict):
            message = json.get("message", response.reason)
            if not message.endswith("."):
                message += "."
            message += " " + ", ".join(value for value in json.get("errors", {}).values())
        elif isinstance(json, str):
            message = json
        raise HordeRequestError(
            message=message,
            code=response.status,
            response_headers=dict(response.headers),
        )
    return await response.json()


async def gather_with_max_concurrent(
    *coros_or_futures: asyncio.Future[_T] | Coroutine[None, None, _T],
    limit: int,
    return_exceptions: bool = False,
) -> list[_T | Exception]:
    """Operates like asyncio.gather() but never runs more than `limit` of the given coroutines at once."""
    pending = list(coros_or_futures)
    running = []
    results = []
    while pending or running:
        while len(running) < limit and pending:
            running.append(asyncio.ensure_future(pending.pop(0)))
        done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            except Exception as error:
                if return_exceptions:
                    results.append(error)
                else:
                    raise error
        running = [task for task in running if not task.done()]
    return results


async def _concurrent_page_items_fetch(
    self,
    url: URL,
    increment_param: str = "page",
    *,
    logger: Logger,
    model: type[AnyModel] | None = None,
    pages: int = 1,
    start_at: int = 1,
    max_concurrent: int = 5,
) -> list[JsonLike | AnyModel]:
    """Fetch items from a paginated API endpoint with a maximum number of concurrent requests."""
    async def fetch_page_items(page_url: str) -> JsonLike:
        result = await json_request(self.session, HTTPMethod.GET, page_url)
        logger.debug(f"Fetched page {page_url}")
        items = result.get("items", [])
        if model:
            return [model.model_validate(item) for item in items]
        return items

    results = await gather_with_max_concurrent(
        *(
            fetch_page_items(url.with_params(**{increment_param: index}))
            for index in range(start_at, start_at + pages)
        ),
        limit=max_concurrent,
    )
    return itertools.chain.from_iterable(results)
