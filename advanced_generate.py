import asyncio
import contextlib
import io
import math
import re
import textwrap
import time
from collections.abc import Generator
from typing import Any, TypedDict

import aiohttp
import discord
from discord import Interaction
from pydantic import ValidationError

from breadcord.helpers import simple_button
from .ai_horde.models.civitai import CivitAIModel, CivitAIModelVersion, ModelType
from .ai_horde.models.general import HordeRequestError
from .ai_horde.models.horde_meta import ActiveModel, GenerationResponse
from .ai_horde.models.image import (
    ControlType,
    ImageGenerationParams,
    ImageGenerationRequest,
    ImageGenerationStatus,
    LoRA,
    Sampler,
    TextualInversion,
    TIPlacement,
)
from .helpers import APIPackage, LongLastingView, fetch_image, report_error, resize_to_match_area


def strip_codeblock(string: str, /) -> str:
    match = re.match(
        r"```(?:json)?(.+)```",
        string,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return string

    lines = match.group(1).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    code = "\n".join(lines)

    return textwrap.dedent(code)


class CustomModelsModal(discord.ui.Modal, title="Custom model"):
    def __init__(self, *args, possible_models: list[ActiveModel], **kwargs):
        super().__init__(*args, **kwargs)
        self.models: list[str] | None = None
        self._possible_models = possible_models

    models_input = discord.ui.TextInput(
        label="Models",
        placeholder="Enter a comma-separated list of model names...",
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        chosen_models: list[str] = [
            model.strip()
            for model in self.models_input.value.split(",")
            if model.strip()
        ]
        possible_models = {model.name.lower(): model.name for model in self._possible_models}
        models = [possible_models.get(model.lower()) for model in chosen_models]

        for model in models:
            if model is None:
                await interaction.response.send_message(
                    (
                        f"Model {model!r} is not available. "
                        f"Models are matched exactly, "
                        f"so make sure that it is spelled and capitalised correctly, and that the model is available."
                    ),
                    ephemeral=True,
                )
                return

        self.models = models
        await interaction.response.defer()


class ModelSelect(discord.ui.Select):
    def __init__(self, available_models: list[ActiveModel], *, modify: ImageGenerationRequest, apis: APIPackage):
        self.to_modify = modify
        self.apis = apis

        self.available_models = [
            model
            for model in sorted(
                available_models,
                key=lambda model: model.queued,
                reverse=True,
            )
            if model.count > 0 and model.type == "image"
        ]

        self.model_groups: dict[str, list[str] | None] = {
            "Stable Cascade": [
                "Stable Cascade 1.0",
            ],
            "SDXL": [
                "AlbedoBase XL (SDXL)",
                "Fustercluck",
                "ICBINP XL",
                "Anime Illust Diffusion XL",
                "Juggernaut XL",
                "Animagine XL",
                "DreamShaper XL",
            ],
            "Anime": [
                "Anything v3",
                "Anything Diffusion",  # v4.0
                "Anything v5",
            ],
            "Pony Diffusion": [
                "Pony Diffusion XL"
            ],
            "Realistic": [
                "ICBINP XL",
            ],
        }

        def get_desc(models: list[str]) -> str:
            long = ", ".join(sorted(models))
            return long if len(long) <= 99 else f"{long[:90]}..."

        super().__init__(
            placeholder="Select models...",
            options=[
                discord.SelectOption(
                    label=group_name,
                    value=group_name,
                    description=get_desc(group_models),
                )
                for group_name, group_models in self.model_groups.items()
            ] + [
                discord.SelectOption(
                    label="Any",
                    value="__any__",
                    description="Any available model can service this request. (default)",
                ),
                discord.SelectOption(
                    label="Custom",
                    value="__custom__",
                    description="Specify a custom list of models.",
                ),
            ],
        )

    async def callback(self, interaction: discord.Interaction):
        match self.values[0]:
            case "__any__":
                self.to_modify.models = None
            case "__custom__":
                modal = CustomModelsModal(possible_models=self.available_models)
                await interaction.response.send_modal(modal)
                await modal.wait()
                self.to_modify.models = modal.models
                # Sending a modal counts as a response, so we can't defer
                await defer_and_edit(interaction, self.to_modify, self.apis, responded_already=True)
                return
            case _:
                self.to_modify.models = self.model_groups[self.values[0]]

        await defer_and_edit(interaction, self.to_modify, self.apis)


class CustomResolutionModal(discord.ui.Modal, title="Custom resolution"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width: int | None = None
        self.height: int | None = None

    width_input = discord.ui.TextInput(
        label="Width",
        placeholder="Enter a width...",
    )
    height_input = discord.ui.TextInput(
        label="Height",
        placeholder="Enter a height...",
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        width = round(int(self.width_input.value) / 64) * 64
        width = max(64, min(3072, width))
        self.width = width

        height = round(int(self.height_input.value) / 64) * 64
        height = max(64, min(3072, height))
        self.height = height

        await interaction.response.defer()


class ResolutionSelect(discord.ui.Select):
    def __init__(self, base_res: int = 512, *, modify: ImageGenerationRequest, apis: APIPackage):
        self.to_modify = modify
        self.apis = apis

        base_options: dict[str, tuple[float, float] | None] = {
            # Normal (512 base)
            "Square": (base_res, base_res),
            "Portrait": (base_res, base_res * 1.5),
            "Landscape": (base_res * 1.5, base_res),
            # Large (1024 base)
            "Square (Large)": (base_res * 2, base_res * 2),
            "Portrait (Large)": resize_to_match_area((2, 3), (base_res*2)**2),
            "Landscape (Large)": resize_to_match_area((3, 2), (base_res*2)**2),

            "Custom": None,
        }

        options = []
        for name, value in base_options.items():
            if value is None:
                options.append(discord.SelectOption(
                    label=name,
                    description="A custom resolution...",
                    value="Custom",
                ))
                continue
            width, height = value
            # Round to nearest 64 multiple
            width = round(width / 64) * 64
            height = round(height / 64) * 64
            gcd = math.gcd(width, height)

            options.append(discord.SelectOption(
                label=name,
                description=f"{width}x{height}: {width // gcd}:{height // gcd}",
                value=f"{width}x{height}",
            ))

        super().__init__(placeholder="Select a resolution...", options=options)

    async def callback(self, interaction: discord.Interaction):
        # This would be much cleaner with goto ;)
        async def inner() -> bool:
            if self.values[0] == "Custom":
                modal = CustomResolutionModal()
                await interaction.response.send_modal(modal)
                responded_already = True
                await modal.wait()
                width, height = modal.width, modal.height
                if width is None or height is None:
                    return responded_already
            else:
                width, height = self.values[0].split("x")
                responded_already = False
            self.to_modify.params.width = int(width)
            self.to_modify.params.height = int(height)
            return responded_already

        await defer_and_edit(interaction, self.to_modify, self.apis, responded_already=await inner())


class BasicOptionsModal(discord.ui.Modal, title="More generation options"):
    def __init__(
        self,
        *args,
        current_request: ImageGenerationRequest,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.current_request = current_request
        self.apis = apis

        self.positive_prompt = discord.ui.TextInput(
            label="Prompt",
            placeholder="Enter a prompt...",
            style=discord.TextStyle.paragraph,
            min_length=1,
            # Max total prompt length (positive + negative) is 1000, we do half - 50 in case something else is injected
            max_length=450,
            default=self.current_request.positive_prompt,
        )
        self.add_item(self.positive_prompt)

        self.negative_prompt = discord.ui.TextInput(
            label="Negative prompt",
            placeholder="Enter a negative prompt...",
            style=discord.TextStyle.paragraph,
            required=False,
            # Max total prompt length (positive + negative) is 1000, we do half - 50 in case something else is injected
            max_length=450,
            default=self.current_request.negative_prompt,
        )
        self.add_item(self.negative_prompt)

        self.seed = discord.ui.TextInput(
            label="Seed",
            placeholder="Enter a seed...",
            required=False,
            default=self.current_request.params.seed,
        )
        self.add_item(self.seed)

        self.steps = discord.ui.TextInput(
            label="Steps - 1 \u2264 steps \u2264 100",
            placeholder="Enter a step count...",
            required=False,
            default=self.current_request.params.steps,
        )
        self.add_item(self.steps)

        self.image_count = discord.ui.TextInput(
            label="Image count - 1 \u2264 images \u2264 10",
            placeholder="Enter an image count...",
            required=False,
            default=self.current_request.params.image_count,
        )
        self.add_item(self.image_count)

    async def on_submit(self, interaction: Interaction, /) -> None:
        self.current_request.positive_prompt = self.positive_prompt.value
        self.current_request.negative_prompt = self.negative_prompt.value or None
        self.current_request.params.seed = self.seed.value or None

        steps = self.steps.value.strip()
        steps = max(1, min(100, int(steps))) if steps else None
        self.current_request.params.steps = steps

        image_count = self.image_count.value.strip()
        image_count = max(1, min(10, int(image_count))) if image_count else None
        self.current_request.params.image_count = image_count

        await defer_and_edit(interaction, self.current_request, self.apis)

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await report_error(interaction, error)


class AdvancedOptionsModal(discord.ui.Modal, title="More generation options"):
    def __init__(
        self,
        *args,
        current_request: ImageGenerationRequest,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.current_request = current_request
        self.apis = apis

        self.cfg_scale = discord.ui.TextInput(
            label="CFG scale - 0 \u2264 cfg \u2264 100",
            placeholder="Enter a CFG scale...",
            required=False,
            default=self.current_request.params.cfg_scale,
        )
        self.add_item(self.cfg_scale)

        self.sampler = discord.ui.TextInput(
            label="Sampler",
            placeholder="Enter a sampler...",
            required=False,
            default=self.current_request.params.sampler,
        )
        self.add_item(self.sampler)

        self.denoising_strength = discord.ui.TextInput(
            label="Denoising strength - 0 \u2264 strength \u2264 100",
            placeholder="Enter a denoising strength...",
            required=False,
            default=self.current_request.params.denoising_strength,
        )
        self.add_item(self.denoising_strength)

        self.clip_skip = discord.ui.TextInput(
            label="Clip skip - 1 \u2264 layers \u2264 12",
            placeholder="Enter a clip skip...",
            required=False,
            default=self.current_request.params.clip_skip,
        )
        self.add_item(self.clip_skip)

    async def on_submit(self, interaction: Interaction, /) -> None:
        responded_already = False

        self.current_request.params.cfg_scale = self._parse_float(self.cfg_scale.value, 0.0, 100.0)
        self.current_request.params.denoising_strength = self._parse_float(self.denoising_strength.value, 0.01, 1.0)
        self.current_request.params.clip_skip = self._parse_int(self.clip_skip.value, 1, 12)

        sampler_value = self.sampler.value.strip().upper().replace(" ", "_")
        if not sampler_value:
            sampler = None
        elif sampler_value in Sampler.__members__:
            sampler = Sampler[sampler_value]
        elif (key := f"K_{sampler_value}") in Sampler.__members__:
            sampler = Sampler[key]
        else:
            sampler = self.sampler.value.strip()
            await interaction.response.send_message(
                "\n".join((
                    f"Sampler {sampler!r} is not a recognised sampler!",
                    f"You may proceed with it, but your request will likely be rejected.",
                )),
                ephemeral=True,
            )
            responded_already = True
        self.current_request.params.sampler = sampler

        await defer_and_edit(interaction, self.current_request, self.apis, responded_already=responded_already)

    @staticmethod
    def _parse_float(value: Any, /, min_value: float, max_value: float) -> float | None:
        value = value.strip()
        return max(min_value, min(max_value, float(value))) if value else None

    @staticmethod
    def _parse_int(value: str, /, min_value: int, max_value: int) -> int | None:
        value = value.strip()
        return max(min_value, min(max_value, int(value))) if value else None

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        try:
            await interaction.response.send_message(str(error), ephemeral=True)
        except discord.InteractionResponded:
            self.apis.logger.exception("Failed to send error message")
            await report_error(interaction, error)
        raise error


class GenerationSettingsView(LongLastingView):
    def __init__(
        self,
        apis: APIPackage,
        default_request: ImageGenerationRequest,
        author_id: int,
    ) -> None:
        super().__init__()
        self.author_id = author_id
        self.apis = apis
        self.logger = apis.logger
        self.cache = apis.cache
        self.horde_api = apis.horde
        self.civitai_api = apis.civitai

        self._default_request = default_request
        self.generation_request = default_request
        self.generation_request.params = self.generation_request.params or ImageGenerationParams()

        self.model_select = ModelSelect(self.cache.horde_models, modify=self.generation_request, apis=self.apis)
        self.add_item(self.model_select)

        self.resolution_select = ResolutionSelect(modify=self.generation_request, apis=self.apis)
        self.add_item(self.resolution_select)

        self._set_nsfw(self.generation_request.nsfw)
        self._set_hires_fix(self.generation_request.params.hires_fix)
        self._set_transparent(self.generation_request.params.transparent)
        # TODO: Put support for post processors somewhere?
        #  Might need to split out lora/ti adding to another message, and also put post processors there.
        #  Remember to deal with facefixer_strength?

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.author_id

    @discord.ui.button(label="Basic options", style=discord.ButtonStyle.blurple, row=2)
    async def more_options(self, interaction: discord.Interaction, _):
        await interaction.response.send_modal(
            BasicOptionsModal(
                current_request=self.generation_request,
                apis=self.apis,
            ),
        )

    @discord.ui.button(label="Advanced options", style=discord.ButtonStyle.blurple, row=2)
    async def even_more_options(self, interaction: discord.Interaction, _):
        await interaction.response.send_modal(
            AdvancedOptionsModal(
                current_request=self.generation_request,
                apis=self.apis,
            ),
        )

    def _set_nsfw(self, value: bool) -> None:
        self.generation_request.nsfw = value
        self.nsfw_toggle.style = discord.ButtonStyle.red if value else discord.ButtonStyle.grey
        self.nsfw_toggle.label = " ".join((*self.nsfw_toggle.label.split(" ")[:-1], "Yes" if value else "No"))

    @discord.ui.button(label="Allow NSFW: No", style=discord.ButtonStyle.grey, row=2)
    async def nsfw_toggle(self, interaction: discord.Interaction, _):
        self._set_nsfw(not self.generation_request.nsfw)
        await defer_and_edit(interaction, self.generation_request, self.apis, view=self)

    @discord.ui.button(label="Source image", style=discord.ButtonStyle.blurple, row=3)
    async def source_image(self, interaction: discord.Interaction, _):
        view = SourceImageView(
            apis=self.apis,
            author_id=self.author_id,
            generation_request=self.generation_request,
        )
        await interaction.response.send_message(
            view=view,
            **await get_source_image_params(self.generation_request, self.apis, send_new=True),
        )
        await view.wait()
        await defer_and_edit(interaction, self.generation_request, self.apis, responded_already=True, update_image=True)

    @discord.ui.button(label="LoRAs and TIs", style=discord.ButtonStyle.blurple, row=3)
    async def modify_loras_or_tis(self, interaction: discord.Interaction, _):
        view = LoRAPickerView(
            apis=self.apis,
            author_id=self.author_id,
            default_loras=self.generation_request.params.loras,
            default_tis=self.generation_request.params.textual_inversions,
        )
        await interaction.response.send_message(
            (
                "Chose LoRAs and TIs. "
                "You can find models at https://civitai.com/models, "
                "filtering for loras and embeddings (here called textual inversions) respectively."
            ),
            view=view,
            embeds=await get_lora_ti_embeds(
                apis=self.apis,
                loras=self.generation_request.params.loras,
                textual_inversions=self.generation_request.params.textual_inversions,
            ),
        )

        # Remove lora and ti embeds form the main message, so it looks like they have moved to the new one
        self.generation_request.params.loras = []
        self.generation_request.params.textual_inversions = []
        await defer_and_edit(interaction, self.generation_request, self.apis, responded_already=True)

        await view.wait()
        self.generation_request.params.loras = view.loras
        self.generation_request.params.textual_inversions = view.textual_inversions
        await defer_and_edit(interaction, self.generation_request, self.apis, responded_already=True)

    def _set_hires_fix(self, value: bool) -> None:
        self.generation_request.params.hires_fix = value
        self.hires_fix_toggle.label = " ".join((*self.hires_fix_toggle.label.split(" ")[:-1], "Yes" if value else "No"))

    @discord.ui.button(label="Hires fix: No", style=discord.ButtonStyle.grey, row=3)
    async def hires_fix_toggle(self, interaction: discord.Interaction, _):
        self._set_hires_fix(not self.generation_request.params.hires_fix)
        await defer_and_edit(interaction, self.generation_request, self.apis, view=self)

    def _set_transparent(self, value: bool) -> None:
        self.generation_request.params.transparent = value
        self.transparency_toggle.label = " ".join((
            *self.transparency_toggle.label.split(" ")[:-1], "Yes" if value else "No"
        ))

    @discord.ui.button(label="Transparent BG: No", style=discord.ButtonStyle.grey, row=3)
    async def transparency_toggle(self, interaction: discord.Interaction, _):
        self._set_transparent(not self.generation_request.params.transparent)
        await defer_and_edit(interaction, self.generation_request, self.apis, view=self)

    @discord.ui.button(label="Generate", style=discord.ButtonStyle.green, row=4, emoji="\N{HEAVY CHECK MARK}")
    async def generate(self, interaction: discord.Interaction, _):
        for item in self.children:
            if item == self.get_json:
                continue
            self.remove_item(item)

        await interaction.message.edit(view=self, content="Generation requested, settings locked.")
        await interaction.response.defer()
        await process_generation(
            interaction,
            self.generation_request.model_copy(),
            apis=self.apis,
            reply_to=interaction.message,
        )

    @discord.ui.button(
        label="Generate & edit",
        style=discord.ButtonStyle.green,
        row=4,
        emoji="\N{HEAVY PLUS SIGN}",
    )
    async def generate_and_continue(self, interaction: discord.Interaction, _):
        await interaction.message.edit(view=self, content="Generation requested, settings locked.")
        await interaction.response.defer()
        await process_generation(
            interaction,
            self.generation_request.model_copy(),
            apis=self.apis,
            reply_to=interaction.message,
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red, row=4, emoji="\N{HEAVY MULTIPLICATION X}")
    async def cancel(self, interaction: discord.Interaction, _):
        await interaction.message.delete()
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="Get JSON", style=discord.ButtonStyle.grey, row=4, emoji="\N{INBOX TRAY}")
    async def get_json(self, interaction: discord.Interaction, _):
        json = self.generation_request.model_dump_json(indent=4)
        try:
            await interaction.response.send_message(
                f"```json\n{json}\n```",
                ephemeral=True,
            )
        except discord.HTTPException:
            small_json = self.generation_request.model_dump_json()
            try:
                await interaction.response.send_message(
                    f"```json\n{small_json}\n```",
                    ephemeral=True,
                )
            except discord.HTTPException:
                await interaction.response.send_message(
                    "The request data is too large to send as a message, so it has been attached as a file.",
                    file=discord.File(
                        io.BytesIO(json.encode()),
                        filename="request.json",
                    ),
                    ephemeral=True,
                )

    @discord.ui.button(label="Load JSON", style=discord.ButtonStyle.grey, row=4, emoji="\N{OUTBOX TRAY}")
    async def load_json(self, interaction: discord.Interaction, _):
        timeout = 30
        await interaction.response.send_message(
            f"Please reply to this message with the JSON data to load either in a codeblock or as a file. \n"
            f"This message will time out <t:{int(time.time() + timeout)}:R>.",
        )
        request_message = await interaction.original_response()

        def reply_check(message: discord.Message) -> bool:
            if message.author.id != self.author_id or message.channel.id != interaction.channel_id:
                return False
            if not message.reference:
                return False
            if message.reference.message_id not in (request_message.id, interaction.message.id):
                return False
            return True

        try:
            reply = await interaction.client.wait_for("message", check=reply_check, timeout=timeout)
        except asyncio.TimeoutError:
            await request_message.delete()
            return

        if reply.attachments:
            async with self.apis.session.get(reply.attachments[0].url) as response:
                json = await response.read()
        else:
            json = strip_codeblock(reply.content)

        try:
            self.generation_request = ImageGenerationRequest.model_validate_json(json)
            self.generation_request.replacement_filter = True
        except ValidationError as error:
            await reply.reply(
                f"Failed to load JSON: "
                f"{discord.utils.escape_markdown(str(error))}",
            )
            self.logger.exception("Failed to load JSON")
            return
        # Update the button
        self._set_nsfw(self.generation_request.nsfw)
        self._set_hires_fix(self.generation_request.params.hires_fix)
        self._set_transparent(self.generation_request.params.transparent)

        await request_message.delete()
        with contextlib.suppress(discord.HTTPException):
            await reply.delete()
        await defer_and_edit(interaction, self.generation_request, self.apis, view=self, responded_already=True)


def model_id_from_url(url: str, /) -> str:
    matches = re.match(r"https?://civitai.com/models/([0-9]+)(?:\?modelVersionId=([0-9]+))?", url)
    if matches is None:
        raise ValueError("Failed to parse CivitAI model URL.")
    model_id, version_id = matches.groups()
    return version_id or model_id


class LoRAPickerModal(discord.ui.Modal, title="LoRA"):
    def __init__(
        self,
        *args,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.apis = apis
        self.loras = []

    identifier = discord.ui.TextInput(
        label="LoRA ID or name",
        placeholder="Enter a CivitAI LoRA model or LoRA version ID, or its exact name...",
        max_length=255,
    )
    model_strength = discord.ui.TextInput(
        label="LoRA model strength - -5 \u2264 strength \u2264 5",
        placeholder="Enter a model strength...",
        required=False,
    )
    clip_strength = discord.ui.TextInput(
        label="LoRA clip strength - -5 \u2264 strength \u2264 5",
        placeholder="Enter a clip strength...",
        required=False,
    )
    is_version = discord.ui.TextInput(
        label="If the LoRA ID is a version ID - True/False",
        placeholder='Enter "True" or "False"...',
        required=False,
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        model_id = self.identifier.value.strip()
        if re.match(r"^https?://civitai.com/models/", model_id):
            model_id = model_id_from_url(model_id)

        self.loras.append(LoRA(
            identifier=model_id,
            strength_model=float(self.model_strength.value) if self.model_strength.value else None,
            strength_clip=float(self.clip_strength.value) if self.clip_strength.value else None,
            is_version=check_truthy(self.is_version.value) if self.is_version.value else None,
        ))
        await interaction.response.defer()

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)


class TextualInversionPickerModal(discord.ui.Modal, title="Textual Inversion"):
    def __init__(
        self,
        *args,
        apis: APIPackage,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.apis = apis
        self.textual_inversions = []

    identifier = discord.ui.TextInput(
        label="Textual Inversion ID or name",
        placeholder="Enter a CivitAI Textual Inversion model ID, or its exact name...",
        max_length=255,
    )
    injection_location = discord.ui.TextInput(
        label="Injection location (prompt/negative prompt)",
        placeholder='Enter "prompt" or "negative prompt"...',
        default="Prompt",
    )
    strength = discord.ui.TextInput(
        label="Textual Inversion strength - -5 \u2264 str \u2264 5",
        placeholder="Enter a strength...",
        required=False,
    )

    async def on_submit(self, interaction: Interaction, /) -> None:
        def transform_location(value: str) -> TIPlacement | None:
            value = value.strip().lower()
            if value.startswith("p"):
                return TIPlacement.PROMPT
            if value.startswith("n"):
                return TIPlacement.NEGATIVE_PROMPT
            return None

        model_id = self.identifier.value.strip()
        if re.match(r"^https?://civitai.com/models/", model_id):
            model_id = model_id_from_url(model_id)

        self.textual_inversions = self.textual_inversions or []
        self.textual_inversions.append(TextualInversion(
            identifier=model_id,
            injection_location=transform_location(self.injection_location.value),
            strength=float(self.strength.value) if self.strength.value else None,
        ))
        await interaction.response.defer()

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)


class LoRAPickerView(LongLastingView):
    def __init__(
        self,
        apis: APIPackage,
        author_id: int,
        default_loras: list[LoRA] | None,
        default_tis: list[TextualInversion] | None,
    ) -> None:
        super().__init__()
        self.author_id = author_id
        self.apis = apis

        self.loras = default_loras or []
        self.textual_inversions = default_tis or []

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.author_id

    @discord.ui.button(label="Add LoRA", style=discord.ButtonStyle.green, row=0)
    async def add_lora(self, interaction: discord.Interaction, _):
        modal = LoRAPickerModal(apis=self.apis)
        await interaction.response.send_modal(modal)
        await modal.wait()
        self.loras.extend(modal.loras)
        new_embeds = [
            await get_lora_embed(lora, self.apis)
            for lora in modal.loras
        ]
        await interaction.message.edit(embeds=interaction.message.embeds + new_embeds)

    @discord.ui.button(label="Remove LoRA", style=discord.ButtonStyle.red, row=0)
    async def remove_lora(self, interaction: discord.Interaction, _):
        if not self.loras:
            await interaction.response.defer()
            return
        self.loras.pop()
        interaction.message.embeds.pop()
        await interaction.message.edit(embeds=interaction.message.embeds)

    @discord.ui.button(label="Add Textual Inversion", style=discord.ButtonStyle.green, row=1)
    async def add_ti(self, interaction: discord.Interaction, _):
        modal = TextualInversionPickerModal(apis=self.apis)
        await interaction.response.send_modal(modal)
        await modal.wait()
        self.textual_inversions.extend(modal.textual_inversions)
        new_embeds = [
            await get_ti_embed(ti, self.apis)
            for ti in modal.textual_inversions
        ]
        await interaction.message.edit(embeds=interaction.message.embeds + new_embeds)

    @discord.ui.button(label="Remove Textual Inversion", style=discord.ButtonStyle.red, row=1)
    async def remove_ti(self, interaction: discord.Interaction, _):
        if not self.textual_inversions:
            await interaction.response.defer()
            return
        self.textual_inversions.pop()
        interaction.message.embeds.pop()
        await interaction.message.edit(embeds=interaction.message.embeds)

    @discord.ui.button(label="Done", style=discord.ButtonStyle.green, row=4, emoji="\N{HEAVY CHECK MARK}")
    async def done(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        await interaction.message.delete()
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red, row=4, emoji="\N{HEAVY MULTIPLICATION X}")
    async def cancel(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        await interaction.message.delete()
        self.stop()


class ControlTypeSelect(discord.ui.Select):
    def __init__(self, generation_request: ImageGenerationRequest) -> None:
        self.generation_request = generation_request

        super().__init__(
            placeholder="Select a control type...",
            options=[
                discord.SelectOption(label="None"),
                *(discord.SelectOption(label=control_type) for control_type in ControlType),
            ],
        )

    async def callback(self, interaction: discord.Interaction):
        self.generation_request.params.control_type = ControlType(self.values[0]) if self.values[0] != "None" else None
        await interaction.response.defer()


class SourceImageParams(TypedDict):
    embed: discord.Embed
    files: list[discord.File] | None
    attachments: list[discord.File] | None


async def files_from_request(request: ImageGenerationRequest, /, session: aiohttp.ClientSession) -> list[discord.File]:
    attachments = []
    if request.source_image:
        attachments.append(discord.File(
            await fetch_image(request.source_image, session=session),
            filename="source_image.webp",
        ))
    for i, extra_source_image in enumerate(request.extra_source_images or []):
        attachments.append(discord.File(
            await fetch_image(extra_source_image, session=session),
            filename=f"extra_source_image_{i}.webp",
        ))
    if request.source_mask:
        attachments.append(discord.File(
            await fetch_image(request.source_mask, session=session),
            filename="source_mask.webp",
        ))
    return attachments


# noinspection PyUnusedLocal
async def get_source_image_params(
    generation_request: ImageGenerationRequest,
    apis: APIPackage,
    send_new: bool = False,
) -> SourceImageParams:
    files = await files_from_request(generation_request, session=apis.session)
    params: SourceImageParams = dict(
        embed=discord.Embed(
            title="Source image",
            description=(
                f"Selected control type: "
                f"{generation_request.params.control_type.value if generation_request.params.control_type else 'None'}"
            ),
        ).set_image(url="attachment://source_image.webp"),
    )
    if send_new:
        params["files"] = files
    else:
        params["attachments"] = files
    return params


class SourceImageView(LongLastingView):
    def __init__(
        self,
        apis: APIPackage,
        author_id: int,
        generation_request: ImageGenerationRequest,
    ) -> None:
        super().__init__()
        self.author_id = author_id
        self.apis = apis
        self.generation_request = generation_request
        # Update buttons to reflect current state
        self.source_image = self.source_image
        self.extra_source_images = self.extra_source_images

        # TODO: If set to none when image_is_control is set, we need to change the button to its off state
        self.control_type_select = ControlTypeSelect(generation_request)
        self.add_item(self.control_type_select)

    @property
    def source_image(self) -> bytes | str | None:
        return self.generation_request.source_image

    @source_image.setter
    def source_image(self, value: bytes | str | None) -> None:
        self.generation_request.source_image = value
        if value is None:
            if self.delete_image in self.children:
                self.remove_item(self.delete_image)
        elif self.delete_image not in self.children:
            self.add_item(self.delete_image)

    @property
    def extra_source_images(self) -> list[bytes | str] | None:
        return self.generation_request.extra_source_images

    @extra_source_images.setter
    def extra_source_images(self, value: list[bytes | str] | None) -> None:
        self.generation_request.extra_source_images = value
        if not value:
            if self.reset_extra_images in self.children:
                self.remove_item(self.reset_extra_images)
            if self.delete_last_extra_image in self.children:
                self.remove_item(self.delete_last_extra_image)
        else:
            if self.reset_extra_images not in self.children:
                self.add_item(self.reset_extra_images)
            if self.delete_last_extra_image not in self.children:
                self.add_item(self.delete_last_extra_image)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.author_id

    async def ask_for_images(
        self, interaction: discord.Interaction, max_images: int | None = None, timeout: int = 15,
    ) -> list[bytes]:
        for button in self.children:
            if button in [self.cancel]:
                continue
            button.disabled = True
        try:
            await interaction.message.edit(
                embed=discord.Embed(
                    title="Awaiting image...",
                    description="\n".join((
                        "Reply to this message with "
                        + ("an image" if (max_images or 2) == 1 else f"up to {max_images} images"),
                        f"Timeout: <t:{int(time.time() + timeout)}:R>",
                    )),
                    colour=discord.Colour.orange(),
                ),
                view=self,
            )
        finally:
            for child in self.children:
                child.disabled = False

        def reply_check(message: discord.Message) -> bool:
            if message.author.id != self.author_id or message.channel.id != interaction.channel_id:
                return False
            if not message.reference or message.reference.message_id != interaction.message.id:
                return False
            return True

        user_reply = await interaction.client.wait_for("message", check=reply_check, timeout=timeout)

        def get_image_urls() -> Generator[str, None, None]:
            for attachment in user_reply.attachments:
                if attachment.content_type.startswith("image/"):
                    yield attachment.proxy_url
            for embed in user_reply.embeds:
                if embed.thumbnail:
                    yield embed.thumbnail.proxy_url
                if embed.image:
                    yield embed.image.proxy_url

        images: list[bytes] = []
        for i, image_url in enumerate(get_image_urls()):
            if max_images is not None and i >= max_images:
                break
            # We need to fetch because the image URL will expire after a while when deleted
            async with self.apis.session.get(image_url) as response:
                if not response.ok:
                    raise RuntimeError(f"Failed to fetch image from {image_url}.")
                images.append(await response.read())
        with contextlib.suppress(discord.HTTPException):
            await user_reply.delete()
        return images

    @discord.ui.button(label="Set source image", style=discord.ButtonStyle.green, row=0)
    async def set_image(self, interaction: discord.Interaction, _):
        await interaction.response.defer()

        try:
            images = await self.ask_for_images(interaction, max_images=1)
        except asyncio.TimeoutError:
            with contextlib.suppress(discord.NotFound):  # Message might have been deleted/canceled
                await interaction.message.edit(
                    view=self,
                    **await get_source_image_params(self.generation_request, self.apis),
                )
            return
        except RuntimeError as error:
            with contextlib.suppress(discord.NotFound):  # Message might have been deleted/canceled
                await interaction.message.edit(
                    embed=discord.Embed(
                        title="Failed to fetch image",
                        description=str(error),
                        colour=discord.Colour.red(),
                    ),
                )
                await asyncio.sleep(5)
                await interaction.message.edit(
                    view=self,
                    **await get_source_image_params(self.generation_request, self.apis),
                )
            return

        self.source_image = images[0]
        await interaction.message.edit(view=self, **await get_source_image_params(self.generation_request, self.apis))

    @discord.ui.button(label="Delete image", style=discord.ButtonStyle.red, row=0)
    async def delete_image(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        self.source_image = None
        await interaction.message.edit(view=self, **await get_source_image_params(self.generation_request, self.apis))

    @discord.ui.button(label='Add "extra" images', style=discord.ButtonStyle.green, row=1)
    async def add_extra_images(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        try:
            images = await self.ask_for_images(interaction, max_images=5, timeout=25)
        except asyncio.TimeoutError:
            with contextlib.suppress(discord.NotFound):
                await interaction.message.edit(
                    view=self,
                    **await get_source_image_params(self.generation_request, self.apis),
                )
            return
        except RuntimeError as error:
            with contextlib.suppress(discord.NotFound):
                await interaction.message.edit(
                    embed=discord.Embed(
                        title="Failed to fetch image",
                        description=str(error),
                        colour=discord.Colour.red(),
                    ),
                )
                await asyncio.sleep(5)
                await interaction.message.edit(
                    view=self,
                    **await get_source_image_params(self.generation_request, self.apis),
                )
            return

        if self.extra_source_images is not None:
            self.extra_source_images.extend(images)
        else:
            self.extra_source_images = images
        await interaction.message.edit(view=self, **await get_source_image_params(self.generation_request, self.apis))

    @discord.ui.button(label="Reset extra images", style=discord.ButtonStyle.red, row=1)
    async def reset_extra_images(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        self.extra_source_images = []
        await interaction.message.edit(view=self, **await get_source_image_params(self.generation_request, self.apis))

    @discord.ui.button(label="Delete last extra image", style=discord.ButtonStyle.red, row=1)
    async def delete_last_extra_image(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        if self.extra_source_images:
            self.extra_source_images.pop()
            await interaction.message.edit(
                view=self,
                **await get_source_image_params(self.generation_request, self.apis),
            )

    @discord.ui.button(label="Image is ControlNet map: No", style=discord.ButtonStyle.red, row=2)
    async def controlnet_image_toggle(self, interaction: discord.Interaction, button: discord.ui.Button):
        if (
            not self.generation_request.params.image_is_control
            and self.generation_request.params.control_type is None
        ):
            await interaction.response.send_message(
                "You must select a control type before you can use the image for ControlNet.",
                ephemeral=True,
            )
            return
        self.generation_request.params.image_is_control = button.style != discord.ButtonStyle.green
        button.style = (
            discord.ButtonStyle.green
            if self.generation_request.params.image_is_control
            else discord.ButtonStyle.red
        )
        button.label = f"Image is ControlNet map: {'Yes' if self.generation_request.params.image_is_control else 'No'}"
        await interaction.response.defer()
        await interaction.message.edit(view=self)

    @discord.ui.button(label="Return control map: No", style=discord.ButtonStyle.red, row=2)
    async def return_control_map_toggle(self, interaction: discord.Interaction, button: discord.ui.Button):
        if (
            not self.generation_request.params.return_control_map
            and self.generation_request.params.control_type is None
        ):
            await interaction.response.send_message(
                "You must select a control type before you can return the control map.",
                ephemeral=True,
            )
            return
        self.generation_request.params.return_control_map = button.style != discord.ButtonStyle.green
        button.style = (
            discord.ButtonStyle.green
            if self.generation_request.params.return_control_map
            else discord.ButtonStyle.red
        )
        button.label = f"Return control map: {'Yes' if self.generation_request.params.return_control_map else 'No'}"
        await interaction.response.defer()
        await interaction.message.edit(view=self)

    @discord.ui.button(label="Done", style=discord.ButtonStyle.green, row=4, emoji="\N{HEAVY CHECK MARK}")
    async def done(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        await interaction.message.delete()
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red, row=4, emoji="\N{HEAVY MULTIPLICATION X}")
    async def cancel(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        await interaction.message.delete()
        self.stop()


async def defer_and_edit(
    interaction: discord.Interaction,
    generation_request: ImageGenerationRequest,
    apis: APIPackage,
    *,
    view: discord.ui.View | None = None,
    responded_already: bool = False,
    update_image: bool = False,
) -> None:
    if not responded_already:
        await interaction.response.defer()

    params = {}
    if update_image:
        files = await files_from_request(generation_request, session=apis.session)
        params["attachments"] = files[:10]

    if view is not None:
        params["view"] = view

    await interaction.message.edit(
        embeds=await get_settings_embeds(generation_request, apis),
        **params,
    )


async def edit_loras_tis(
    interaction: discord.Interaction,
    apis: APIPackage,
    *,
    loras: list[LoRA],
    textual_inversions: list[TextualInversion],
) -> None:
    embeds = await get_lora_ti_embeds(apis=apis, loras=loras, textual_inversions=textual_inversions)
    await interaction.message.edit(embeds=embeds[-10:])


async def get_settings_embeds(generation_request: ImageGenerationRequest, apis: APIPackage) -> list[discord.Embed]:
    description = []

    def append_truthy(key_name: str, value: Any):
        if value:
            description.append(f"**{key_name}:** {value}")

    append_truthy("Prompt", generation_request.positive_prompt)
    append_truthy("Negative prompt", generation_request.negative_prompt)
    append_truthy("Seed", generation_request.params.seed)
    append_truthy("Models", ", ".join(generation_request.models or ["Any"]))
    append_truthy(
        "Resolution",
        f"{generation_request.params.width or 512}x{generation_request.params.height or 512}",
    )
    append_truthy("Steps", generation_request.params.steps)
    append_truthy("CFG scale", generation_request.params.cfg_scale)
    append_truthy("Image count", generation_request.params.image_count or 1)
    append_truthy(
        "Control type",
        generation_request.params.control_type if generation_request.params.control_type else None,
    )
    append_truthy("Denoising strength", generation_request.params.denoising_strength)
    append_truthy("Sampler", generation_request.params.sampler or None)

    embeds = [
        discord.Embed(
            title="Generation settings",
            description="\n".join(description),
            colour=discord.Colour.blurple(),
        )
        .set_footer(text="Click the buttons below to modify the request.")
        .set_thumbnail(url="attachment://source_image.webp"),
    ]

    lora_ti_embeds = await get_lora_ti_embeds(
        apis=apis,
        loras=generation_request.params.loras,
        textual_inversions=generation_request.params.textual_inversions,
    )
    if len(lora_ti_embeds) < 10 - len(embeds):
        embeds.extend(lora_ti_embeds)
    else:
        embeds.append(discord.Embed(
            title="LoRAs and TIs",
            description="Too many LoRAs and TIs to display.",
            colour=discord.Colour.blurple(),
        ))
    return embeds


async def get_finished_embed(
    generation_request: ImageGenerationRequest,
    finished_generation: ImageGenerationStatus,
    apis: APIPackage,
) -> list[discord.Embed]:
    description = []

    def append_truthy(key_name: str, value: Any):
        if value:
            description.append(f"**{key_name}:** {value}")

    append_truthy("Prompt", generation_request.positive_prompt)
    append_truthy("Negative prompt", generation_request.negative_prompt)
    append_truthy("Base seed", generation_request.params.seed)
    if len(individual_seeds := {gen.seed for gen in finished_generation.generations}) > 1:
        append_truthy("Individual seeds", ", ".join(individual_seeds))
    append_truthy("NSFW", generation_request.nsfw)
    append_truthy(
        ("Models" if len(finished_generation.generations) > 1 else "Model"),
        ", ".join({
            generation.model
            for generation in finished_generation.generations
        })
    )
    append_truthy(
        "Resolution",
        f"{generation_request.params.width or 512}x{generation_request.params.height or 512}",
    )
    append_truthy("Steps", generation_request.params.steps)
    append_truthy("CFG scale", generation_request.params.cfg_scale)
    append_truthy(
        "Image count",
        f"{finished_generation.finished}/{generation_request.params.image_count or 1}",
    )
    append_truthy("Denoising strength", generation_request.params.denoising_strength)
    append_truthy("Sampler", sampler.value if (sampler := generation_request.params.sampler) else None)
    append_truthy("Hires Fix", generation_request.params.hires_fix)
    append_truthy("LoRAs", ", ".join(lora.identifier for lora in generation_request.params.loras or []))
    append_truthy(
        "Textual inversions",
        ", ".join(ti.identifier for ti in generation_request.params.textual_inversions or []),
    )
    description.append("")

    finished_workers = {
        (generation.worker_name, generation.worker_id)
        for generation in finished_generation.generations
    }
    if len(finished_workers) == 1:
        worker_name, worker_id = next(iter(finished_workers))
        append_truthy("Finished by", f"{worker_name} (`{worker_id}`)")
    else:
        append_truthy("Finished by", "".join(
            f"\n{worker_name} (`{worker_id}`)"
            for worker_name, worker_id in finished_workers
        ))

    was_censored = len(finished_generation.generations) <= 1 and finished_generation.generations[0].censored
    embeds = [
        discord.Embed(
            title="Generation finished" if not was_censored else "Generation censored",
            description="\n".join(description),
            colour=discord.Colour.green() if not was_censored else discord.Colour.red(),
        ),
    ]

    lora_ti_embeds = await get_lora_ti_embeds(
        apis=apis,
        loras=generation_request.params.loras,
        textual_inversions=generation_request.params.textual_inversions,
    )
    if len(lora_ti_embeds) < 10 - len(embeds):
        embeds.extend(lora_ti_embeds)
    else:
        embeds.append(discord.Embed(
            title="LoRAs and TIs",
            description="Too many LoRAs and TIs to display.",
            colour=discord.Colour.blurple(),
        ))
    return embeds


async def get_lora_ti_embeds(
    *,
    apis: APIPackage,
    loras: list[LoRA] | None,
    textual_inversions: list[TextualInversion] | None,
) -> list[discord.Embed]:
    # TODO: Make use cache
    embeds = []
    for lora in loras or []:
        embeds.append(await get_lora_embed(lora, apis))
    for textual_inversion in textual_inversions or []:
        embeds.append(await get_ti_embed(textual_inversion, apis))

    return embeds


async def get_lora_embed(lora: LoRA, apis: APIPackage) -> discord.Embed:
    civitai_model: CivitAIModel | CivitAIModelVersion | None
    if lora.is_version:
        civitai_model = await apis.civitai.get_model_version(lora.identifier)
        images = civitai_model.images if civitai_model else []
    else:
        civitai_model = await apis.civitai.get_model(lora.identifier)
        images = civitai_model.versions[0].images if civitai_model.versions else []

    if civitai_model and civitai_model.type != ModelType.LORA:
        raise ValueError(f"Model {civitai_model.identifier!r} is not a LoRA model.")

    if civitai_model:
        return discord.Embed(
            title=f"LoRA: {civitai_model.name}",
            url=f"https://civitai.com/models/{civitai_model.id}",
            description="\n".join((
                f"**ID:** {civitai_model.id} {'(version)' if lora.is_version else ''}",
                f"**Strength (model):** {lora.strength_model}",
                f"**Strength (clip):** {lora.strength_clip}",
            )),
            colour=discord.Colour.orange(),
        ).set_thumbnail(url=next((image.url for image in images if not image.nsfw), None))
    else:
        return discord.Embed(
            title=f"LoRA: {lora.identifier}",
            description="\n".join((
                f"**Strength (model):** {lora.strength_model}",
                f"**Strength (clip):** {lora.strength_clip}",
            )),
            colour=discord.Colour.orange(),
        )


async def get_ti_embed(textual_inversion: TextualInversion, apis: APIPackage) -> discord.Embed:
    civitai_model = await apis.civitai.get_model(textual_inversion.identifier)
    if civitai_model and civitai_model.type != ModelType.TEXTUALINVERSION:
        raise ValueError(f"Model {textual_inversion.identifier!r} is not a Textual Inversion model.")

    if civitai_model:
        return discord.Embed(
            title=f"Textual inversion: {civitai_model.name}",
            url=f"https://civitai.com/models/{civitai_model.id}",
            description="\n".join((
                f"**ID:** {civitai_model.id}",
                f"**Strength:** {textual_inversion.strength}",
                f"**Injection location:** {textual_inversion.injection_location}",
            )),
            colour=discord.Colour.blue(),
        ).set_thumbnail(url=civitai_model.versions[0].images[0].url)
    else:
        return discord.Embed(
            title=f"Textual inversion: {textual_inversion.identifier}",
            description="\n".join((
                f"**Strength:** {textual_inversion.strength}",
                f"**Injection location:** {textual_inversion.injection_location}",
            )),
            colour=discord.Colour.blue(),
        )


def check_truthy(value: str) -> bool | None:
    value = value.strip().lower()
    if value.startswith(("t", "y")):
        return True
    if value.startswith(("f", "n")):
        return False
    return None


class AttachmentDeletionView(discord.ui.View):
    def __init__(self, required_votes: int = 2) -> None:
        super().__init__(timeout=None)
        self.required_votes = required_votes
        self.already_voted: set[int] = set()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id not in self.already_voted

    @simple_button(label="Delete", style=discord.ButtonStyle.red)
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        needed_votes = self.required_votes
        if interaction.guild is None:
            needed_votes = 1
        elif interaction.guild.member_count and interaction.guild.member_count < 8:
            needed_votes = 1

        self.already_voted.add(interaction.user.id)
        button.label = f"Delete ({len(self.already_voted)}/{needed_votes})"
        if len(self.already_voted) < needed_votes:
            return
        await interaction.response.defer()
        lobotomised_view = self
        for child in self.children:
            if child == button:
                button.disabled = True
        await interaction.message.edit(attachments=[], view=lobotomised_view)


class DeleteOrRetryView(AttachmentDeletionView):
    def __init__(
        self,
        *,
        generation_params: ImageGenerationRequest,
        finished_generation_status: ImageGenerationStatus,
        apis: APIPackage,
        required_votes: int = 1,
    ) -> None:
        super().__init__(required_votes=required_votes)
        self.generation_params = generation_params
        self.finished_generation_status = finished_generation_status
        self.apis = apis

    @discord.ui.button(
        label="Retry", style=discord.ButtonStyle.blurple,
        emoji="\N{CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS}",
    )
    async def retry(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        await process_generation(
            interaction=interaction,
            generation_request=self.generation_params,
            apis=self.apis,
            reply_to=interaction.message,
            edit=any(generation.censored for generation in self.finished_generation_status.generations),
        )

    @discord.ui.button(
        label="Edit", style=discord.ButtonStyle.blurple,
        emoji="\N{CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS}",
    )
    async def retry_and_edit(self, interaction: discord.Interaction, _):
        await interaction.response.send_message(
            "Choose generation settings",
            view=GenerationSettingsView(
                apis=self.apis,
                default_request=self.generation_params,
                author_id=interaction.user.id,
            ),
            embeds=await get_settings_embeds(self.generation_params, self.apis),
        )


class CancelGenerationView(discord.ui.View):
    def __init__(self, author_id: int, apis: APIPackage, generation: GenerationResponse) -> None:
        super().__init__()
        self.author_id = author_id
        self.apis = apis
        self.generation = generation
        self.canceled = False

    async def interaction_check(self, interaction: Interaction, /) -> bool:
        return interaction.user.id == self.author_id

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red)
    async def cancel_generation(self, interaction: discord.Interaction, _):
        self.canceled = True
        await interaction.response.defer()
        await self.apis.horde.cancel_image_generation(generation_id=self.generation.id)
        await interaction.message.delete()
        self.stop()


async def process_generation(
    interaction: discord.Interaction,
    generation_request: ImageGenerationRequest,
    *,
    apis: APIPackage,
    reply_to: discord.Message,
    edit: bool = False,
) -> discord.Message | None:
    start_time = time.time()
    user_account = await apis.horde.get_current_user()
    is_anon = user_account.id == 0
    try:
        queued_generation = await apis.horde.queue_image_generation(generation_request)
    except HordeRequestError as error:
        if error.code == 403:
            embed = discord.Embed(
                title="You do not have the required kudos to queue this generation",
                description=(
                    f"{error}\n\n"
                ) + ("**This might be solved by logging in to the horde using `/horde login`**" if is_anon else ""),
                colour=discord.Colour.red(),
            )
            return await interaction.followup.send(embed=embed, ephemeral=True)

        await report_error(interaction, error)
        apis.logger.exception("Failed to queue image generation")
        return None

    requested_images = generation_request.params.image_count or 1
    generic_wait_message = "\n".join((
        f"Please wait while your generation is being processed.",
        f"Generation ID: `{queued_generation.id}`\n\n"
    ))
    embed = discord.Embed(
        title="Generating...",
        description=generic_wait_message,
        colour=discord.Colour.blurple(),
    ).set_footer(text=f"Using {'anonymous' if is_anon else 'logged in'} account.")

    view = CancelGenerationView(
        author_id=interaction.user.id,
        apis=apis,
        generation=queued_generation,
    )
    if edit:
        await reply_to.edit(embed=embed, attachments=[], view=view)
        message = reply_to
    else:
        message = await reply_to.reply(embed=embed, view=view)

    await asyncio.sleep(8)

    time_between_checks = 5
    last_finished = 0
    while True:
        if view.canceled:
            return message

        generation_check = await apis.horde.get_image_generation_check(queued_generation.id)
        if generation_check.done:
            break

        params = {}
        if last_finished < generation_check.finished < requested_images:
            full_generations = (await apis.horde.get_image_generation_status(queued_generation.id)).generations
            params["attachments"] = [
                discord.File(
                    fp=await fetch_image(generation.img, apis.session),
                    filename=f"{generation.id}.webp",
                )
                for generation in full_generations
            ]
            last_finished = generation_check.finished

        estimated_done_at = int(time.time() + max(generation_check.wait_time, time_between_checks + 1) + 1)
        embed.description = (
            f"{generic_wait_message}"
            + (f"Generated {generation_check.finished}/{requested_images} images. \n" if requested_images > 1 else "")
            + f"Estimated to be done <t:{estimated_done_at}:R>."
        )
        await message.edit(embed=embed, **params)

        # Generations time out after 10 minutes
        if time.time() - start_time > 60 * 10:
            await message.edit(embed=discord.Embed(
                title="Generation timed out.",
                description="Please try again, or try a different model.",
                colour=discord.Colour.red(),
            ))
            return message

        await asyncio.sleep(time_between_checks)

    generation_status = await apis.horde.get_image_generation_status(queued_generation.id)

    embeds = await get_finished_embed(generation_request, generation_status, apis)
    embeds[0].set_footer(text=f"Time taken: {round(time.time() - start_time, 2)}s")

    await message.edit(
        embeds=embeds,
        attachments=[
            discord.File(
                fp=await fetch_image(generation.img, apis.session),
                filename=f"{generation.id}.webp",
            )
            for generation in generation_status.generations
        ] if not (len(generation_status.generations) == 1 and generation_status.generations[0].censored) else [],
        view=DeleteOrRetryView(
            required_votes=2,
            generation_params=generation_request,
            finished_generation_status=generation_status,
            apis=apis,
        ),
    )
