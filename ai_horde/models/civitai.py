import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, ConfigDict, AliasChoices

from .general import HordeModel, HordeSuccess


# noinspection SpellCheckingInspection
class ModelType(Enum):
    CHECKPOINT = "Checkpoint"
    TEXTUALINVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    AESTHETICGRADIENT = "AestheticGradient"
    LORA = "LORA"
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


class CivitAIImageMetadata(HordeModel):
    prompt: str | None = Field(
        default=None,
        description="The image's prompt.",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="The image's negative prompt.",
        validation_alias=AliasChoices("negativePrompt", "negative_prompt"),
    )


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
        validation_alias="meta"
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
    fp: Literal["fp16", "fp32"] | None = Field(
        default=None,
        description="The model file's floating point precision.",
    )
    size: Literal["full", "pruned"] | None = Field(
        default=None,
        description="The model's size (full or pruned).",
    )
    format: CivitAIFileFormat | None = Field(
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
        description="The model version's description. Usually a changelog.",
    )
    created_at: datetime.datetime = Field(
        description="The model version's creation date.",
        validation_alias="createdAt",
    )
    download_url: str = Field(
        description="The model version's download URL.",
        validation_alias="downloadUrl",
    )
    trainedWords: list[str] = Field(
        description="The words used to trigger the model.",
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
        description="The model description.",
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
