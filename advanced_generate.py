import asyncio
import io
import math
import time
from logging import Logger
from typing import Any, NamedTuple

import aiohttp
import discord
from discord import Interaction

from .ai_horde.cache import Cache
from .ai_horde.interface import CivitAIAPI, HordeAPI
from .ai_horde.models.civitai import CivitAIModel, CivitAIModelVersion, ModelType
from .ai_horde.models.horde_meta import ActiveModel
from .ai_horde.models.image import (
    Base64Image,
    ImageGenerationParams,
    ImageGenerationRequest,
    ImageGenerationStatus,
    LoRA,
    Sampler,
    TextualInversion,
    TIPlacement,
)


class APIPackage(NamedTuple):
    horde: HordeAPI
    civitai: CivitAIAPI
    cache: Cache
    logger: Logger
    session: aiohttp.ClientSession


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
        models = [
            model.strip()
            for model in self.models_input.value.split(",")
            if model.strip()
        ]
        possible_models = tuple(model.name for model in self._possible_models)
        for model in models:
            if model not in possible_models:
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

        model_groups: dict[str, list[str] | None] = {
            "SDXL": [
                "Fustercluck",
                "ICBINP XL",
                "AlbedoBase XL (SDXL)",
            ],
            "Anime": [
                "Anything v3",
                "Anything Diffusion",  # v4.0
                "Anything v5",
            ],
            "Realistic": [
                "ICBINP XL",
            ],
        }

        super().__init__(
            placeholder="Select a model...",
            options=[
                discord.SelectOption(
                    label=group_name,
                    value=",".join(group_models),
                    description=", ".join(sorted(group_models)),
                )
                for group_name, group_models in model_groups.items()
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
                self.to_modify.models = self.values[0].split(",")

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
            "Portrait (Large)": (base_res * 2, base_res * 3),
            "Landscape (Large)": (base_res * 3, base_res * 2),

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
        await interaction.response.send_message(str(error), ephemeral=True)


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

    async def on_submit(self, interaction: Interaction, /) -> None:
        cfg_scale = self.cfg_scale.value.strip()
        cfg_scale = max(0.0, min(100.0, float(cfg_scale))) if cfg_scale else None
        self.current_request.params.cfg_scale = cfg_scale

        if (key := self.sampler.value.strip().upper()) in Sampler.__members__:
            sampler = Sampler[key]
        else:
            sampler = self.sampler.value.strip()
        self.current_request.params.sampler = sampler

        denoising_strength = self.denoising_strength.value.strip()
        denoising_strength = max(0.01, min(1.0, float(denoising_strength))) if denoising_strength else None
        self.current_request.params.denoising_strength = denoising_strength

        await defer_and_edit(interaction, self.current_request, self.apis)

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)
        raise error


class GenerationSettingsView(discord.ui.View):
    def __init__(
        self,
        apis: APIPackage,
        default_request: ImageGenerationRequest,
        author_id: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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

    @discord.ui.button(label="Allow NSFW: No", style=discord.ButtonStyle.red, row=2)
    async def nsfw_toggle(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.generation_request.nsfw = button.style != discord.ButtonStyle.green
        button.style = discord.ButtonStyle.green if self.generation_request.nsfw else discord.ButtonStyle.red
        button.label = f"Allow NSFW: {'Yes' if self.generation_request.nsfw else 'No'}"
        await interaction.response.defer()
        await interaction.message.edit(view=self, embeds=await get_settings_embed(self.generation_request, self.apis))

    @discord.ui.button(label="Change LoRAs and TIs", style=discord.ButtonStyle.green, row=2)
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

    @discord.ui.button(label="Generate", style=discord.ButtonStyle.green, row=4)
    async def generate(self, interaction: discord.Interaction, _):
        for item in self.children:
            if hasattr(item, "disabled"):
                item.disabled = True
            if hasattr(item, "enabled"):
                item.enabled = False

        await interaction.response.defer()
        await interaction.message.edit(view=self)
        await process_generation(self.generation_request, apis=self.apis, reply_to=interaction.message)

    @discord.ui.button(label="Get JSON", style=discord.ButtonStyle.gray, row=4, emoji="\N{PAGE FACING UP}")
    async def get_json(self, interaction: discord.Interaction, _):
        json = self.generation_request.model_dump_json(indent=4, exclude_none=True)
        await interaction.response.send_message(
            f"AI horde request data: ```json\n{json}\n```",
            ephemeral=True,
        )
        # TODO: Allow inputting arbitrary json (through a new message asking for a reply?), and have it validated


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
        self.loras.append(LoRA(
            identifier=self.identifier.value.strip(),
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

        self.textual_inversions = self.textual_inversions or []
        self.textual_inversions.append(TextualInversion(
            identifier=self.identifier.value.strip(),
            injection_location=transform_location(self.injection_location.value),
            strength=float(self.strength.value) if self.strength.value else None,
        ))
        await interaction.response.defer()

    async def on_error(self, interaction: Interaction, error: Exception, /) -> None:
        await interaction.response.send_message(str(error), ephemeral=True)


class LoRAPickerView(discord.ui.View):
    def __init__(
        self,
        apis: APIPackage,
        author_id: int,
        default_loras: list[LoRA] | None,
        default_tis: list[TextualInversion] | None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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

    @discord.ui.button(label="Done", style=discord.ButtonStyle.gray, row=4)
    async def done(self, interaction: discord.Interaction, _):
        await interaction.response.defer()
        await interaction.message.delete()
        self.stop()


async def defer_and_edit(
    interaction: discord.Interaction,
    generation_request: ImageGenerationRequest,
    apis: APIPackage,
    *,
    responded_already: bool = False,
) -> None:
    if not responded_already:
        await interaction.response.defer()
    await interaction.message.edit(embeds=await get_settings_embed(generation_request, apis))


async def edit_loras_tis(
    interaction: discord.Interaction,
    apis: APIPackage,
    *,
    loras: list[LoRA],
    textual_inversions: list[TextualInversion],
) -> None:
    embeds = await get_lora_ti_embeds(apis=apis, loras=loras, textual_inversions=textual_inversions)
    await interaction.message.edit(embeds=embeds[-10:])


async def get_settings_embed(generation_request: ImageGenerationRequest, apis: APIPackage) -> list[discord.Embed]:
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
    append_truthy("Denoising strength", generation_request.params.denoising_strength)

    embeds = [
        discord.Embed(
            title="Generation settings",
            description="\n".join(description),
            colour=discord.Colour.blurple(),
        ).set_footer(text="Click the buttons below to modify the request."),
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
    if len(individual_seeds := [gen.seed for gen in finished_generation.generations]) > 1:
        append_truthy("Individual seeds", ", ".join(individual_seeds))
    append_truthy("NSFW", generation_request.nsfw)
    append_truthy("Models", ", ".join({
        generation.model
        for generation in finished_generation.generations
    }))
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
    append_truthy("LoRAs", ", ".join(lora.identifier for lora in generation_request.params.loras or []))
    append_truthy(
        "Textual inversions",
        ", ".join(ti.identifier for ti in generation_request.params.textual_inversions or []),
    )
    description.append("\n")
    append_truthy("Finished by", ", ".join({
        generation.worker_id
        for generation in finished_generation.generations
    }))

    embeds = [
        discord.Embed(
            title="Generation finished",
            description="\n".join(description),
            colour=discord.Colour.green(),
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
    def __init__(
        self,
        *args,
        required_votes: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.required_votes = required_votes
        self.already_voted: set[int] = set()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id not in self.already_voted

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.red)
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.already_voted.add(interaction.user.id)
        button.label = f"Delete ({len(self.already_voted)}/{self.required_votes})"
        if len(self.already_voted) < self.required_votes:
            return
        await interaction.response.defer()
        await interaction.message.edit(view=None, attachments=[])


async def process_generation(
    generation_request: ImageGenerationRequest,
    *,
    apis: APIPackage,
    reply_to: discord.Message,
) -> discord.Message:
    start_time = time.time()
    queued_generation = await apis.horde.queue_image_generation(generation_request)

    generic_wait_message = (
        f"Please wait while your generation is being processed.\n"
        f"Generation ID: {queued_generation.id}\n\n"
    )
    embed = discord.Embed(
        title="Generating...",
        description=generic_wait_message,
        colour=discord.Colour.blurple(),
    )
    message = await reply_to.reply(embed=embed)

    await asyncio.sleep(5)
    finished_images = 0
    requested_images = generation_request.params.image_count or 1

    while True:
        generation_check = await apis.horde.get_generation_status(queued_generation.id)

        if generation_check.finished != finished_images and not generation_check.done:
            finished_images = generation_check.finished
            embed.description = (
                f"{generic_wait_message}"
                f"Generated {finished_images}/{requested_images} images."
            )
            await message.edit(embed=embed)

        if generation_check.done:
            break

        # Generations time out after 10 minutes
        if time.time() - start_time > 60 * 10:
            await message.edit(embed=discord.Embed(
                title="Generation timed out.",
                description="Please try again, or try a different model.",
                colour=discord.Colour.red(),
            ))
            return message

        await asyncio.sleep(5)

    generation_status = await apis.horde.get_generation_status(queued_generation.id, full=True)

    embeds = await get_finished_embed(generation_request, generation_status, apis)
    embeds[0].set_footer(text=f"Time taken: {round(time.time() - start_time, 2)}s")

    async def fetch_image(image: Base64Image | str) -> io.BytesIO:
        if isinstance(image, Base64Image):
            return image.to_bytesio()
        async with apis.session.get(image) as response:
            return io.BytesIO(await response.read())

    await message.edit(
        embeds=embeds,
        attachments=[
            discord.File(
                fp=await fetch_image(generation.img),
                filename=f"{generation.id}.webp",
            )
            for generation in generation_status.generations
        ],
        view=AttachmentDeletionView(required_votes=2),
    )
    return None
