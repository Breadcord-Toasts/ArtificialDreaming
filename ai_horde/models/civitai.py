import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator

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
    # TODO: Rename to "metadata"
    meta: dict | None = Field(
        default=None,
        description="The image's metadata.",
    )

    # noinspection PyNestedDecorators
    @field_validator("nsfw", mode="before")
    @classmethod
    def validate_nsfw(cls, value: str) -> NSFWLevel | None:
        if value == "None" or value is None:
            return None
        return NSFWLevel(value)


class CivitAICreator(HordeSuccess):
    username: str = Field(
        description="The creator's username. This acts as a unique identifier.",
    )
    # TODO: Rename to "avatar_url" or "image_url"
    image: str | None = Field(
        default=None,  # TODO: Should this need to be explicitly specified as being None?
        description="The creator's image (avatar).",
    )
    # TODO: Rename to "model_count"
    modelCount: int | None = Field(
        default=None,
        description="The number of models the creator has uploaded.",
    )
    # TODO rename to "models_url" or "models_link"
    link: str | None = Field(
        default=None,
        description="The creator's link.",
    )


class CivitAIModelFileMetadata(HordeModel):
    fp: Literal["fp16", "fp32"] | None = Field(
        default=None,
        description="The model file's floating point precision.",
    )
    # TODO: Make more descriptive, could be confused with "size_kb"
    size: Literal["full", "pruned"] | None = Field(
        default=None,
        description="The model file's size.",
    )
    format: CivitAIFileFormat | None = Field(
        description="The model file's format.",
    )


class CivitAIModelFile(HordeModel):
    # TODO: Rename to "size_kb" (maybe drop the "_kb" suffix? prob not)
    sizeKb: int | None = Field(
        default=None,
        description="The file's size in kilobytes.",
    )
    # TODO: Rename to "pickle_scan_result"
    pickleScanResult: ScanResult = Field(
        description="The status of the pickle scan.",
    )
    # TODO: Rename to "virus_scan_result"
    virusScanResult: ScanResult = Field(
        description="The status of the virus scan.",
    )
    # TODO: rename to "scanned_at"
    # TODO: Cast date string count to datetime.date when deserializing
    scannedAt: datetime.datetime | None = Field(
        description="The date the file was scanned.",
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
    # TODO: Rename to "created_at"
    # TODO: Cast date string count to datetime.date when deserializing.
    #  How in gods name is this meant ot be parsed "2022-11-30T01:14:36.498Z"? Might need a helper function.
    createdAt: datetime.datetime = Field(
        description="The model version's creation date.",
    )
    # TODO: Rename to "download_url"
    downloadUrl: str = Field(
        description="The model version's download URL.",
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
    # TODO: alias to a more descriptive name?
    # TODO: Maybe replace literals with an enum?
    mode: Literal["Archived", "TakenDown"] | None = Field(
        default=None,
        description=(
            "The model's current mode. "
            "If it's archived, the files field will be empty. "
            "If it's taken down, the images field will be empty."
        )
    )
    creator: CivitAICreator = Field(
        description="The model's creator/uploader.",  # TODO: Factcheck "uploader"
    )
    # TODO rename to "model_versions"
    modelVersions: list[CivitAIModelVersion] = Field(
        description="The model's versions.",
    )
