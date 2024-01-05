import asyncio
from http import HTTPMethod
from json import loads as json_loads
from logging import Logger
from typing import Self, Any
from urllib.parse import urlparse, parse_qsl, urlencode

import aiohttp
from pydantic import BaseModel

from .models.civitai import CivitAIModel, ModelType
from .models.general import HordeRequestError, HordeRequest
from .models.horde_meta import ActiveModel, HordeNews, HordeUser, Team, Worker
from .models.image import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationStatus,
    ImageGenerationCheck,
    InterrogationRequest,
    InterrogationStatus,
    InterrogationResponse,
    InterrogationStatusState,
)


class URL(str):
    def __truediv__(self, other: str) -> Self:
        return self.__class__(f"{self.removesuffix('/')}/{other.removeprefix('/')}")

    @property
    def query_params(self) -> dict[str, str]:
        return dict(parse_qsl(urlparse(self).query))

    def clear_params(self) -> Self:
        return self.__class__(urlparse(self)._replace(query="").geturl())

    def set_params(self, **params: Any) -> Self:
        url = urlparse(self)
        query_params = dict(parse_qsl(url.query))
        query_params.update(params)
        return self.__class__(url._replace(query=urlencode(query_params)).geturl())


HORDE_API_BASE = URL("https://stablehorde.net/api/")
CIVITAI_API_DOMAIN = URL("https://civitai.com/api/")

JsonLike = dict[str, "JsonLike"] | list["JsonLike"] | str | int | float | bool | None


class HordeAPI:
    def __init__(self, session: aiohttp.ClientSession, *, logger: Logger):
        self.session = session
        self.logger = logger

    async def generate_image(
        self, generation_settings: ImageGenerationRequest, /
    ) -> ImageGenerationStatus:
        """Simple helper function to both queue an image generation and wait for it to finish."""
        generation = await self.queue_image_generation(generation_settings)

        # There is just about 0 chance that the generation will already be done
        await asyncio.sleep(5)

        while True:
            check = await self.get_generation_status(generation.id)
            if check.done:
                return await self.get_generation_status(generation.id, full=True)
            await asyncio.sleep(3)

    async def queue_image_generation(
        self, generation_settings: ImageGenerationRequest, /
    ) -> ImageGenerationResponse:
        queued_generation = ImageGenerationResponse.model_validate(await json_request(
            self.session, HTTPMethod.POST, HORDE_API_BASE / "v2/generate/async",
            data=generation_settings
        ))
        self.logger.debug(f"Image generation queued: {queued_generation.id}")
        return queued_generation

    async def get_generation_status(
        self, generation_id: str, *, full: bool = False
    ) -> ImageGenerationStatus | ImageGenerationCheck:
        """Get the status of an image generation. A "full" request will contain generated images."""
        json = await json_request(
            self.session,
            HTTPMethod.GET,
            url=HORDE_API_BASE / "v2/generate/" / ("status" if full else "check") / generation_id,
        )
        return ImageGenerationStatus.model_validate(json) if full else ImageGenerationCheck.model_validate(json)

    async def cancel_image_generation(self, generation_id: str) -> None:
        await json_request(self.session, HTTPMethod.DELETE, HORDE_API_BASE / "v2/generate/status" / generation_id)
        return None

    async def interrogate(
        self, interrogation_settings: InterrogationRequest, /
    ) -> InterrogationStatus:
        """Simple helper function to both queue an interrogation and wait for it to finish."""
        interrogation = await self.queue_interrogation(interrogation_settings)

        while True:
            check = await self.get_interrogation_status(interrogation.id)
            if check.state == InterrogationStatusState.DONE:
                return check
            # TODO: Decrease (alter depending on how many are done?)
            await asyncio.sleep(10)

    async def queue_interrogation(
        self, interrogation_settings: InterrogationRequest, /
    ) -> InterrogationResponse:
        queued_interrogation = InterrogationResponse.model_validate(await json_request(
            self.session, HTTPMethod.POST, HORDE_API_BASE / "v2/interrogate/async",
            data=interrogation_settings
        ))
        self.logger.debug(f"Interrogation queued: {queued_interrogation.id}")
        return queued_interrogation

    async def get_interrogation_status(self, interrogation_id: str) -> InterrogationStatus:
        return InterrogationStatus.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/interrogate/status" / interrogation_id
        ))

    async def cancel_interrogation(self, interrogation_id: str) -> None:
        await json_request(self.session, HTTPMethod.DELETE, HORDE_API_BASE / "v2/interrogate/status" / interrogation_id)
        return None

    async def get_models(self) -> list[ActiveModel]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/status/models")
        return [ActiveModel.model_validate(model) for model in json]

    # TODO: Figure out why this is not working, test case: "Anything Diffusion
    # async def get_model(self, model_name: str) -> ActiveModel | None:
    #     json = await json_request(self.session, "GET", HORDE_API_BASE / f"v2/status/models/{model_name}")
    #     if len(json) == 0:
    #         return None
    #     return ActiveModel.model_validate(json[0])

    async def get_users(self) -> list[HordeUser]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/users")
        return [HordeUser.model_validate(user) for user in json]

    async def get_user(self, user_id: int) -> HordeUser:
        return HordeUser.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/users/{user_id}"
        ))

    async def get_current_user(self) -> HordeUser:
        return HordeUser.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/find_user"
        ))

    async def get_teams(self) -> list[Team]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/teams")
        return [Team.model_validate(team) for team in json]

    async def get_team(self, team_id: str) -> Team:
        return Team.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/teams/{team_id}"
        ))

    async def get_workers(self) -> list[Worker]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/workers")
        return [Worker.model_validate(worker) for worker in json]

    async def get_worker(self, worker_id: str) -> Worker:
        return Worker.model_validate(await json_request(
            self.session, HTTPMethod.GET, HORDE_API_BASE / f"v2/workers/{worker_id}"
        ))

    async def get_news(self) -> list[HordeNews]:
        json = await json_request(self.session, HTTPMethod.GET, HORDE_API_BASE / "v2/status/news")
        return [HordeNews.model_validate(news) for news in json]


class CivitAIAPI:
    def __init__(self, session: aiohttp.ClientSession, *, logger: Logger):
        self.session = session
        self.logger = logger

    async def _fetch_paginated_json_list(self, url: URL, *, pages: int = 1) -> list[JsonLike]:
        total_pages = -1
        items = []
        for page in range(1, pages + 1):
            if total_pages != -1 and page > total_pages:
                break
            url = url.set_params(page=page)
            self.logger.debug(f"Fetching page {page} of a requested {pages}.")
            json = await json_request(self.session, HTTPMethod.GET, url)
            total_pages = json.get("metadata", {}).get("totalPages", total_pages)
            page_items = json.get("items", [])
            if not page_items:
                break
            items.extend(page_items)

        return items

    # noinspection PyShadowingBuiltins
    async def get_models(
        self,
        *,
        pages: int = 1,
        type: ModelType | None = None,
    ) -> list[CivitAIModel]:
        url = CIVITAI_API_DOMAIN / "v1/models"
        if type is not None:
            url = url.set_params(types=type.value)

        models = await self._fetch_paginated_json_list(
            url,
            pages=pages,
        )
        return [CivitAIModel.model_validate(model) for model in models]


async def json_request(
    session: aiohttp.ClientSession,
    method: HTTPMethod,
    url: str,
    *,
    data: dict[str, JsonLike] | BaseModel | None = None,
) -> JsonLike:
    """
    Helper function to make a request with a pydantic model as the data

    :param session: aiohttp.ClientSession to use
    :param method: HTTP method to use
    :param url: URL to make the request to
    :param data: Data to send, either a pydantic model or a dict that can be serialised to JSON
    :return: The JSON response as a dict
    """
    if isinstance(data, HordeRequest):
        response = await session.request(
            method,
            url,
            # This is a bit of a hack, but not sure if there's a better way.
            json=json_loads(data.model_dump_json(exclude_unset=True, by_alias=True)),
        )
    else:
        response = await session.request(method, url, json=data)
    if not response.ok:
        message = (await response.json()).get("message")
        raise HordeRequestError(
            message=message,
            code=response.status,
            response_headers=dict(response.headers),
        )
    return await response.json()
