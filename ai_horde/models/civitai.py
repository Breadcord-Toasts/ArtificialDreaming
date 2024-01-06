import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, ConfigDict, Field, field_validator, model_validator

from .general import HordeModel, HordeSuccess


# noinspection SpellCheckingInspection
class ModelType(Enum):
    CHECKPOINT = "Checkpoint"
    TEXTUALINVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    AESTHETICGRADIENT = "AestheticGradient"
    LORA = "LORA"
    LOCON = "LoCon"
    CONTROLNET = "Controlnet"
    POSES = "Poses"


class ScanResult(Enum):
    PENDING = "Pending"
    SUCCESS = "Success"
    DANGER = "Danger"
    ERROR = "Error"


# noinspection SpellCheckingInspection
class CivitAIFileFormat(Enum):
    SAFETENSOR = "SafeTensor"
    PICKLETENSOR = "PickleTensor"
    OTHER = "Other"


class NSFWLevel(Enum):
    MATURE = "Mature"
    SOFT = "Soft"
    X = "X"


def validate_dimensions(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    assert isinstance(value, str), "Must be a string"
    split = value.split("x", maxsplit=1)
    assert len(split) == 2, 'Must be in the format "widthxheight"'
    return int(split[0]), int(split[1])


Dimensions = Annotated[tuple[int, int], BeforeValidator(validate_dimensions)]


class CivitAIImageMetadata(HordeModel):
    """Generation parameters for a CivitAI image. Further fields will likely be added at runtime."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    # Fields are so dynamic that it's not a nice UX to raise an exception if one doesn't exist
    def __getattr__(self, item: str) -> Any:
        try:
            return super().__getattr__(item)
        except AttributeError:
            return None

    prompt: str | None = Field(default=None)
    negative_prompt: str | None = Field(default=None)
    size: Dimensions | None = Field(default=None)
    hires_resize: Dimensions | None = Field(default=None)
    seed: int | None = Field(default=None)
    steps: int | None = Field(default=None)
    sampler: str | None = Field(default=None)
    cfg_scale: float | None = Field(default=None)
    clip_skip: int | None = Field(default=None)
    hires_steps: int | None = Field(default=None)
    denoising_strength: float | None = Field(default=None)

    model: str | None = Field(default=None)
    vae: str | None = Field(default=None)
    hashes: dict[str, str] | None = Field(default=None)

    resources: list[dict[str, Any]] | None = Field(default=None)

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def fix_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        def fix_stupid_key_name(key: str) -> str:
            key = key.replace(" ", "_")
            snake_case_key = ""
            for index, char in enumerate(key):
                if char.isupper() and index != 0 and len(key) > index + 1 and not key[index + 1].isupper():
                    snake_case_key += "_"
                snake_case_key += char.lower()
            return snake_case_key

        return {fix_stupid_key_name(key): value for key, value in data.items()}


class CivitAIImage(HordeSuccess):
    url: str = Field(
        description="The image's URL.",
    )
    width: int = Field(
        description="The image's width.",
    )
    height: int = Field(
        description="The image's height.",
    )
    nsfw: NSFWLevel | None = Field(
        description="The image's NSFW level.",
    )
    hash: str = Field(
        description="The image's hash.",
    )
    # TODO: They just didn't document this so I need to figure it out myself :bruhmike:
    metadata: CivitAIImageMetadata | None = Field(
        default=None,
        description="The image's metadata, like generation parameters.",
        validation_alias="meta",
    )

    # noinspection PyNestedDecorators
    @field_validator("nsfw", mode="before")
    @classmethod
    def validate_nsfw(cls, value: str) -> NSFWLevel | None:
        if value == "None" or value is None:
            return None
        return NSFWLevel(value)


class CivitAICreator(HordeSuccess):
    # Removes a warning about field names starting with "model_" being reserved
    model_config = ConfigDict(protected_namespaces=())

    username: str = Field(
        description="The creator's username. This acts as a unique identifier.",
    )
    image_url: str | None = Field(
        default=None,
        description="The creator's image (avatar).",
        validation_alias="image",
    )
    model_count: int | None = Field(
        default=None,
        description="The number of models the creator has uploaded.",
        validation_alias="modelCount",
    )
    uploaded_models_url: str | None = Field(
        default=None,
        description="Link to the creator's uploaded models.",
        validation_alias="link",
    )


class CivitAIModelFileMetadata(HordeModel):
    fp: str | None = Field(
        default=None,
        description="The model file's floating point precision, such as fp16 or fp32",
    )
    size: Literal["full", "pruned"] | None = Field(
        default=None,
        description="The model's size (full or pruned).",
    )
    format: CivitAIFileFormat | None = Field(
        default=None,
        description="The model file's format.",
    )


class CivitAIModelFile(HordeModel):
    size_kb: int | None = Field(
        default=None,
        description="The file's size in kilobytes.",
        validation_alias="sizeKb",
    )
    pickle_scan: ScanResult = Field(
        description="The status of the pickle scan.",
        validation_alias="pickleScanResult",
    )
    virus_scan: ScanResult = Field(
        description="The status of the virus scan.",
        validation_alias="virusScanResult",
    )
    scanned_at: datetime.datetime | None = Field(
        description="The date the file was scanned.",
        validation_alias="scannedAt",
    )
    metadata: CivitAIModelFileMetadata = Field(
        description="The file's metadata.",
    )


class CivitAIModelVersion(HordeModel):
    id: int = Field(
        description="The model version's identifier.",
    )
    name: str = Field(
        description="The model version's name.",
    )
    description: str | None = Field(
        default=None,
        description="The model version's description as HTML. Usually a changelog.",
    )
    created_at: datetime.datetime = Field(
        description="The model version's creation date.",
        validation_alias="createdAt",
    )
    download_url: str = Field(
        description="The model version's download URL.",
        validation_alias="downloadUrl",
    )
    trained_words: list[str] = Field(
        description="The words used to trigger the model.",
        validation_alias="trainedWords",
    )
    files: list[CivitAIModelFile] = Field(
        description="The model version's files.",
    )
    images: list[CivitAIImage] = Field(
        description="The model version's images.",
    )


class CivitAIModel(HordeSuccess):
    id: int = Field(
        description="THe model identifier.",
    )
    name: str = Field(
        description="The model name.",
    )
    description: str = Field(
        description="The model description as HTML.",
    )
    type: ModelType = Field(
        description="The model type.",
    )
    nsfw: bool = Field(
        description="Whether the model is NSFW or not.",
    )
    tags: list[str] = Field(
        description="The model's associated tags.",
    )
    state: Literal["Archived", "TakenDown"] | None = Field(
        default=None,
        description=(
            "The model's current mode. "
            "If it's archived, the files field will be empty. "
            "If it's taken down, the images field will be empty."
        ),
        validation_alias="mode",
    )
    creator: CivitAICreator = Field(
        description="The model's creator.",
    )
    versions: list[CivitAIModelVersion] = Field(
        default=[],  # TODO: Guh??? Something is wrong. Check the json
        description="The model's versions.",
        validation_alias="modelVersions",
    )
