from typing import Any

from pydantic import Field, model_validator

from .general import HordeModel
from .image import LoRA, Sampler, TextualInversion


class Style(HordeModel):
    name: str = Field(
        min_length=1,
        description="The name of the style.",
    )

    positive_prompt: str = Field(
        min_length=1,
        description="The positive prompt which will be used to generate the image(s).",
        pattern=r".*\{positive_prompt\}.*",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="The negative prompt which will be used to generate the image(s).",
        pattern=r".*\{negative_prompt\}.*",
    )

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def fix_prompt(cls, data: Any) -> Any:
        assert isinstance(data, dict), "Data must be a dict"
        if "prompt" not in data:
            return data

        prompt = data.pop("prompt")
        assert isinstance(prompt, str), "Prompt must be a string"
        prompt = prompt.replace("{p}", "{positive_prompt}")
        prompt = prompt.replace("{np}", "{negative_prompt}")

        if "###" not in prompt and "{negative_prompt}" in prompt:
            prompt = prompt.replace("{negative_prompt}", "###{negative_prompt}")

        prompts = last_split(prompt, "###")
        data["positive_prompt"] = prompts[0]
        data["negative_prompt"] = prompts[1] if len(prompts) > 1 else None

        return data

    model: str | None = Field(
        default=None,
        description="The model to use for the image generation.",
    )
    steps: int | None = Field(
        default=None,
        description="The number of sampling steps to perform when generating this request. ",
        ge=1, le=500,
    )
    width: int | None = Field(
        default=None,
        description="The width of the image to generate",
        ge=64, le=3072, multiple_of=64,
    )
    height: int | None = Field(
        default=None,
        description="The height of the image to generate",
        ge=64, le=3072, multiple_of=64,
    )
    cfg_scale: float | None = Field(
        default=None,
        description=(
            "The CFG scale (classifier-free guidance scale) to use when generating the request. "
            "Higher values makes the image follow the prompt more closely."
        ),
        ge=0.0, le=100.0,
    )
    sampler_name: Sampler | None = Field(
        default=None,
        description="The sampler to use when generating this request.",
    )
    loras: list[LoRA] | None = Field(
        default=None,
        description="A list of LoRAs to use when generating this request.",
    )
    clip_skip: int | None = Field(
        default=None,
        description="The number of CLIP language processor layers to skip.",
        ge=1, le=12,
    )
    hires_fix: bool | None = Field(
        default=None,
        description="If true, process the image at base resolution before upscaling and re-processing.",
    )
    karras: bool | None = Field(
        default=None,
        description="If true, enable karras noise scheduling tweaks.",
    )
    textual_inversions: list[TextualInversion] | None = Field(
        default=None,
        description="A list of Textual Inversions to use when generating this request.",
        validation_alias="tis",
    )


StyleCategory = dict[str, list[str]]


def last_split(string: str, sep: str) -> list[str]:
    """Split a string by a separator, and return the text after the last instance of the separator, and the rest."""
    split = string.split(sep)
    if len(split) <= 2:
        return split

    return [sep.join(split[:-1]), split[-1]]
