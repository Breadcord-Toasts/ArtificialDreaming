import asyncio
import json
import time
from collections.abc import Mapping
from logging import Logger
from pathlib import Path
from typing import Any, Iterable, TypeVar

import aiofiles
import aiohttp
from pydantic import BaseModel, ValidationError

from .interface import URL, CivitAIAPI, HordeAPI, JsonLike
from .models.civitai import CivitAIModel
from .models.horde_meta import ActiveModel
from .models.other_sources import ModelReference, Style, StyleEnhancement

__all__ = (
    "Cache",
)


STYLES_LIST = URL("https://api.github.com/repos/Haidra-Org/AI-Horde-Styles/contents/styles.json")
ENHANCEMENTS_LIST = URL("https://api.github.com/repos/Haidra-Org/AI-Horde-Styles/contents/enhancements.json")
STYLE_CATEGORY_LIST = URL("https://api.github.com/repos/Haidra-Org/AI-Horde-Styles/contents/categories.json")
IMAGE_MODEL_REFERENCE_LIST = URL(
    "https://api.github.com/repos/Haidra-Org/AI-Horde-image-model-reference/contents/stable_diffusion.json",
)

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
        formatted_cache: bool = False,
        invalidation_seconds: int = 60 * 60 * 6,
    ):
        self.session = session
        self.horde = horde_api
        self.civitai = civitai_api

        self.logger = logger
        self.storage_path = storage_path
        self.formatted_logs = formatted_cache
        self.invalidation_seconds = invalidation_seconds

        self.styles: list[Style] | None = None
        self._styles_file = self.storage_path / "styles.json"
        self.enhancements: dict[str, StyleEnhancement] | None = None
        self._enhancements_file = self.storage_path / "enhancements.json"
        self.style_categories: dict[str, list[str]] | None = None
        self._style_categories_file = self.storage_path / "style_categories.json"
        # TODO: Make this cover both image and text models, currently it's just image models
        self.horde_model_reference: dict[str, ModelReference] | None = None
        self._horde_model_reference_file = self.storage_path / "model_reference.json"
        self.horde_models: list[ActiveModel] | None = None
        self._horde_models_file = self.storage_path / "horde_models.json"
        self.civitai_models: list[CivitAIModel] | None = None
        self._civitai_models_file = self.storage_path / "civitai_models.json"

    async def update(self) -> None:
        self.logger.info("Updating cache...")
        await asyncio.gather(
            self.update_styles(),
            self.update_enhancements(),
            self.update_horde_model_reference(),
            self.update_horde_models(),
            self.update_civitai_models(),
        )
        # For things that depend on other things
        await asyncio.gather(
            self.update_style_categories(),
        )

        self.logger.info("Cache updated.")

    async def update_styles(self) -> None:
        if not self.file_outdated(self._styles_file):
            try:
                if not self.styles:
                    self.styles = await self.load_cache(self._styles_file, model=Style)
                return
            except Exception as error:
                self.logger.error(f"Failed to load styles from cache, fetching from API instead. ({error})")

        self.logger.info("Fetching styles...")
        raw_styles = await fetch_github_json_file(self.session, STYLES_LIST)
        raw_styles = [dict(name=name, **style) for name, style in raw_styles.items()]
        self.styles, errors = self._validate_dicts(raw_styles, Style)

        if not errors:
            await self._open_and_dump(self._styles_file, [
                json.loads(style.model_dump_json(by_alias=False))
                for style in self.styles
            ])
        else:
            self.logger.warning(f"Failed to validate {len(errors)} styles, not saving to cache.")

    async def update_enhancements(self) -> None:
        if not self.file_outdated(self._enhancements_file):
            try:
                if not self.enhancements:
                    self.enhancements = await self.load_cache(self._enhancements_file, model=StyleEnhancement)
                return
            except Exception as error:
                self.logger.error(f"Failed to load enhancements from cache, fetching from API instead. ({error})")

        self.logger.info("Fetching enhancements...")
        raw_enhancements: dict[str, dict[str, Any]] = await fetch_github_json_file(self.session, ENHANCEMENTS_LIST)
        self.enhancements, errors = self._validate_dict_dicts(raw_enhancements, StyleEnhancement)

        if not errors:
            await self._open_and_dump(self._enhancements_file, {
                key: json.loads(enhancement.model_dump_json(by_alias=False))
                for key, enhancement in self.enhancements.items()
            })
        else:
            self.logger.warning(f"Failed to validate {len(errors)} enhancements, not saving to cache.")

    async def update_style_categories(self) -> None:
        if not self.file_outdated(self._style_categories_file):
            try:
                if not self.style_categories:
                    self.style_categories = await self.load_cache(self._style_categories_file)
                return
            except Exception as error:
                self.logger.error(f"Failed to load style categories from cache, fetching from API instead. ({error})")

        self.logger.info("Fetching style categories...")
        self.style_categories: dict[str, list[str]] = await fetch_github_json_file(self.session, STYLE_CATEGORY_LIST)
        original_categories = self.style_categories.copy()
        valid_references = [*(style.name for style in self.styles), *original_categories.keys()]
        for category, references in self.style_categories.items():
            for reference in references:
                if reference not in valid_references:
                    self.logger.warning(
                        f"Style {reference} not found in styles list, removing from category {category}"
                    )
                    references.remove(reference)
            self.style_categories[category] = references

        if len(self.style_categories) == len(original_categories):
            await self._open_and_dump(self._style_categories_file, self.style_categories)
        else:
            self.logger.warning("Failed to validate some style categories, not saving to cache.")

    async def update_horde_model_reference(self) -> None:
        if not self.file_outdated(self._horde_model_reference_file):
            try:
                if not self.horde_model_reference:
                    self.horde_model_reference = await self.load_cache(
                        self._horde_model_reference_file,
                        model=ModelReference,
                    )
                return
            except Exception as error:
                self.logger.error(f"Failed to load horde model reference from cache, fetching from API instead. "
                                  f"({error})")

        self.logger.info("Fetching horde model reference...")
        raw_reference = await fetch_github_json_file(self.session, IMAGE_MODEL_REFERENCE_LIST)
        self.horde_model_reference, errors = self._validate_dict_dicts(raw_reference, ModelReference)

        if not errors:
            await self._open_and_dump(self._horde_model_reference_file, {
                key: json.loads(reference.model_dump_json(by_alias=False))
                for key, reference in self.horde_model_reference.items()
            })
        else:
            self.logger.warning(f"Failed to validate {len(errors)} models, not saving to cache.")

    async def update_horde_models(self) -> None:
        if not self.file_outdated(self._horde_models_file):
            try:
                if not self.horde_models:
                    self.horde_models = [
                        ActiveModel.model_validate(model)
                        for model in await self.load_cache(self._horde_models_file)
                    ]
                return
            except Exception as error:
                self.logger.error(f"Failed to load horde models from cache, fetching from API instead. ({error})")

        self.logger.info("Fetching horde models...")
        self.horde_models = await self.horde.get_models()
        await self._open_and_dump(self._horde_models_file, [
            json.loads(model.model_dump_json(by_alias=False))
            for model in self.horde_models
        ])

    async def update_civitai_models(self) -> None:
        if not self.file_outdated(self._civitai_models_file):
            try:
                if not self.civitai_models:
                    self.civitai_models = [
                        CivitAIModel.model_validate(model)
                        for model in await self.load_cache(self._civitai_models_file)
                    ]
                return
            except Exception as error:
                self.logger.error(f"Failed to load CivitAI models from cache, fetching from API instead. ({error})")

        self.logger.info("Fetching CivitAI models...")
        self.civitai_models = await self.civitai.get_models(limit=500)  # TODO: Don't hardcode this number
        # noinspection PyArgumentEqualDefault
        await self._open_and_dump(self._civitai_models_file, [
            json.loads(model.model_dump_json(by_alias=False, exclude_none=True))
            for model in self.civitai_models
        ])

    async def load_cache(
        self,
        path: Path,
        *,
        model: type[BaseModel] | None = None,
        strict: bool = False,
    ) -> JsonLike | BaseModel:
        self.logger.debug(f"Loading cache file {path}")
        async with aiofiles.open(path, encoding="utf-8") as file:
            data = json.loads(await file.read())
            if model is not None:
                if isinstance(data, list):
                    return [model.model_validate(item, strict=strict) for item in data]
                if isinstance(data, dict):
                    return {
                        key: model.model_validate(value, strict=strict)
                        for key, value in data.items()
                    }
            return data

    async def _open_and_dump(self, path: Path, data: JsonLike) -> None:
        # Done first so the file isn't created if the dump fails
        dumped = json.dumps(data, indent=4*self.formatted_logs)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as file:
            await file.write(dumped)

    def _validate_dicts(
        self,
        data: Iterable[JsonLike],
        model: type[AnyModel],
        *,
        strict: bool = False,
    ) -> tuple[list[AnyModel], list[Exception]]:
        validated = []
        errors = []
        for item in data:
            try:
                validated.append(model.model_validate(item, strict=strict))
            except ValidationError as error:
                self.logger.exception(f"Failed to validate {model.__name__} {item}")
                errors.append(error)
        return validated, errors

    def _validate_dict_dicts(
        self,
        data: Mapping[str, JsonLike],
        model: type[AnyModel],
        *,
        strict: bool = False,
    ) -> tuple[dict[str, AnyModel], list[Exception]]:
        validated = {}
        errors = []
        for key, item in data.items():
            try:
                validated[key] = model.model_validate(item, strict=strict)
            except ValidationError as error:
                self.logger.exception(f"Failed to validate {model.__name__} {item}")
                errors.append(error)
        return validated, errors

    def file_outdated(self, path: Path) -> bool:
        if not path.is_file():
            return True

        last_modified = path.stat().st_mtime
        now = time.time()
        return now - last_modified > self.invalidation_seconds


async def fetch_github_json_file(session: aiohttp.ClientSession, url: URL) -> JsonLike:
    response = await session.get(url, headers={"Accept": "application/vnd.github.raw+json"})
    return await response.json()
