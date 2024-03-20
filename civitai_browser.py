import re
from abc import ABC, abstractmethod
from typing import Any

import discord

from .ai_horde.cache import Cache
from .ai_horde.models.civitai import CivitAIModel, ModelType
from .helpers import LongLastingView


class _Unset:
    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<unset>"

    def __str__(self) -> str:
        return "<unset>"


UNSET = _Unset()


class Page:
    def __init__(
        self,
        content: str | _Unset = UNSET,
        embeds: list[discord.Embed] | _Unset = UNSET,
        attachments: list[discord.Attachment | discord.File] | _Unset = UNSET,
    ) -> None:
        self.content = content
        self.embeds = embeds
        self.attachments = attachments

    def unpack(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not UNSET}


class PaginatedView(LongLastingView, ABC):
    def __init__(self, data: list, *, starting_index: int = 0) -> None:
        super().__init__()
        self.data = data
        self.index = starting_index

    @discord.ui.button(
        label="Previous",
        style=discord.ButtonStyle.grey,
        emoji="\N{BLACK LEFT-POINTING TRIANGLE}",
        disabled=True,
    )
    async def previous_page(self, interaction: discord.Interaction, _) -> None:
        self.index -= 1
        self.update_buttons()
        await self.update_page(interaction)

    @discord.ui.button(
        label="Next",
        style=discord.ButtonStyle.grey,
        emoji="\N{BLACK RIGHT-POINTING TRIANGLE}",
    )
    async def next_page(self, interaction: discord.Interaction, _) -> None:
        self.index += 1
        self.update_buttons()
        await self.update_page(interaction)

    def update_buttons(self) -> None:
        self.previous_page.disabled = self.index <= 0
        self.next_page.disabled = self.index >= len(self.data) - 1

    async def update_page(self, interaction: discord.Interaction) -> None:
        await interaction.response.edit_message(view=self, **(await self.get_page()).unpack())

    @abstractmethod
    async def get_page(self) -> Page:
        ...


class CivitAIModelBrowserView(PaginatedView):
    MAX_DESCRIPTION_LENGTH = 2048
    MAX_VERSIONS = 5
    MAX_VERSIONS_CHARS = 500
    MAX_TAGS = 8
    MAX_TAGS_CHARS = 100

    def __init__(self, models: list[CivitAIModel], *, starting_index: int = 0, cache: Cache) -> None:
        super().__init__(data=models, starting_index=starting_index)
        self.cache = cache

    async def get_page(self) -> Page:
        model: CivitAIModel = self.data[self.index]

        on_horde = False
        if horde_model_reference := self.cache.horde_model_reference:
            for horde_model in horde_model_reference.values():
                if not horde_model.homepage or "civitai" not in model.url.lower():
                    continue
                if not (match := re.match(r"^https://civitai\.com/models/(\d+)", horde_model.homepage)):
                    continue
                if model.id == int(match.group(1)):
                    on_horde = True
                    break

        embed = discord.Embed(
            title=model.name,
            description="\n".join(s for s in (
                f"**Model type:** {model.type}",
                f"**Tags:** {self._get_tags_str(model)}",
                f"**NSFW:** {model.nsfw}",
                f"**Available on the horde:** {on_horde}" if model.type == ModelType.CHECKPOINT else None,
            ) if s),
            url=model.url,
            colour=discord.Colour.random(seed=model.id),
        )
        embed.set_image(url=next((image.url for image in model.versions[0].images if not image.nsfw), None))
        embed.set_footer(text=f"Model {self.index + 1}/{len(self.data)}")
        if creator := model.creator:
            embed.set_author(name=model.creator.username, icon_url=creator.image_url, url=creator.uploaded_models_url)

        embed.add_field(
            name="Versions",
            value=self._get_versions_str(model),
        )

        embed.add_field(
            name=((stats := model.stats) and False) or "Stats",
            value="\n".join(s for s in (
                f"**Downloads:** {stats.download_count:,}",
                f"**Rating:** {stats.rating} ({stats.ratingCount:,} ratings)" if stats.rating is not None else "",
            ) if s),
            inline=False,
        )

        file = model.versions[0].files[0]
        if file.size_kb > 1024 ** 2:
            appropriate_filesize = f"{file.size_kb / (1024 * 1024):.2f} GB"
        elif file.size_kb > 1024:
            appropriate_filesize = f"{file.size_kb / 1024:.2f} MB"
        else:
            appropriate_filesize = f"{file.size_kb:.2f} KB"
        embed.add_field(
            name="File",
            value="\n".join(s for s in (
                f"[{file.name}]({file.download_url})",
                f"**Size:** {appropriate_filesize}",
                f"**Type:** {file.type}",
            ) if s),
            inline=False,
        )

        return Page(embeds=[embed])

    def _get_versions_str(self, model: CivitAIModel) -> str:
        string = ", ".join(
            f"[{version.name}]({version.url})"
            for version in model.versions[:self.MAX_VERSIONS]
        )
        if len(string) > self.MAX_VERSIONS_CHARS:
            string = f"{string[:self.MAX_VERSIONS_CHARS]}... and {len(model.versions) - self.MAX_VERSIONS} more"
        return string

    def _get_tags_str(self, model: CivitAIModel) -> str:
        string = ", ".join(model.tags[:self.MAX_TAGS])
        if len(string) > self.MAX_TAGS_CHARS:
            string = f"{string[:self.MAX_TAGS_CHARS]}..."
        return string

