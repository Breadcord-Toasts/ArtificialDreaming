import asyncio
import json
import time
from logging import Logger
from pathlib import Path
from typing import Iterable, TypeVar

import aiofiles
import aiohttp
from pydantic import BaseModel, ValidationError

from .interface import JsonLike, URL, CivitAIAPI, HordeAPI
from .models.civitai import CivitAIModel
from .models.styles import Style, StyleCategory

__all__ = (
    "Cache",
)


STYLES_LIST = URL("https://api.github.com/repos/Haidra-Org/AI-Horde-Styles/contents/styles.json")
STYLE_CATEGORY_LIST = URL("https://api.github.com/repos/Haidra-Org/AI-Horde-Styles/contents/categories.json")

AnyModel = TypeVar("AnyModel", bound=BaseModel)


class Cache:
    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        horde_api: HordeAPI,
        civitai_api: CivitAIAPI,
        logger: Logger,
        storage_path: Path,
        formatted_cache: bool = False
    ):
        self.session = session
        self.horde = horde_api
        self.civitai = civitai_api

        self.logger = logger
        self.storage_path = storage_path
        self.formatted_logs = formatted_cache

        self.styles: list[Style] = []
        self.style_categories: list[StyleCategory] = []
        self.models: list[CivitAIModel] = []

        self._styles_file = self.storage_path / "styles.json"
        self._style_categories_file = self.storage_path / "style_categories.json"
        self._models_file = self.storage_path / "models.json"

    async def update(self) -> None:
        self.logger.info("Updating cache...")
        await asyncio.gather(
            self.update_styles(),
            self.update_style_categories(),
            self.update_models(),
        )
        self.logger.info("Cache updated.")

    async def update_styles(self) -> None:
        if not file_outdated(self._styles_file):
            if not self.styles:
                self.styles = await self.load_cache(self._styles_file, model=Style)
            return

        self.logger.info("Fetching styles...")
        raw_styles = await fetch_github_json_file(self.session, STYLES_LIST)
        raw_styles = [dict(name=name, **style) for name, style in raw_styles.items()]
        self.styles, errors = self._validate_dicts(raw_styles, Style)

        if not errors:
            await self._open_and_dump(self._styles_file, [
                json.loads(style.model_dump_json(by_alias=False))
                for style in self.styles
            ])

    async def update_style_categories(self) -> None:
        if not file_outdated(self._style_categories_file):
            if not self.style_categories:
                self.style_categories = await self.load_cache(self._style_categories_file)
            return

        self.logger.info("Fetching style categories...")
        self.style_categories = await fetch_github_json_file(self.session, STYLE_CATEGORY_LIST)
        await self._open_and_dump(self._style_categories_file, self.style_categories)

    async def update_models(self) -> None:
        if not file_outdated(self._models_file):
            if not self.models:
                self.models = [
                    CivitAIModel.model_validate(model)
                    for model in await self.load_cache(self._models_file)
                ]
            return

        self.logger.info("Fetching models...")
        self.models = await self.civitai.get_models(pages=1)
        await self._open_and_dump(self._models_file, [
            json.loads(model.model_dump_json(by_alias=False))
            for model in self.models
        ])

    async def load_cache(self, path: Path, *, model: type[BaseModel] | None = None) -> JsonLike | BaseModel:
        self.logger.debug(f"Loading cache file {path}")
        async with aiofiles.open(path, "r", encoding="utf-8") as file:
            data = json.loads(await file.read())
            if model is not None:
                return [model.model_validate(item) for item in data]
            return data

    async def _open_and_dump(self, path: Path, data: JsonLike) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as file:
            await file.write(json.dumps(data, indent=4*self.formatted_logs))

    def _validate_dicts(
        self,
        data: Iterable[JsonLike],
        model: type[AnyModel],
    ) -> tuple[list[AnyModel], list[Exception]]:
        validated = []
        errors = []
        for item in data:
            try:
                validated.append(model.model_validate(item))
            except ValidationError as error:
                self.logger.exception(f"Failed to validate {model.__name__} {item}")
                errors.append(error)
        return validated, errors


def file_outdated(path: Path) -> bool:
    if not path.is_file():
        return True

    last_modified = path.stat().st_mtime
    now = time.time()
    return now - last_modified > 60 * 60 * 2


async def fetch_github_json_file(session: aiohttp.ClientSession, url: URL) -> JsonLike:
    response = await session.get(url, headers=dict(Accept="application/vnd.github.raw+json"))
    return await response.json()

