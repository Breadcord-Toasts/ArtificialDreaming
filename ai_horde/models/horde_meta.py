from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator

from .general import HordeModel, HordeSuccess


class TinyActiveModel(HordeModel):
    name: str = Field(
        description="The model name.",
    )
    type: Literal["text", "image"] | None = Field(
        default=None,
        description="The model type.",
    )


class ActiveModel(TinyActiveModel, HordeSuccess):
    count: int = Field(
        description="The number of workers currently serving this model.",
    )
    performance: float = Field(
        description="The average generation speed of this model.",
    )
    eta: int = Field(
        description="Estimated time in seconds for this model's queue to be cleared.",
    )
    jobs: int = Field(
        description="The number of jobs currently waiting to be generated by this model.",
    )
    queued: int = Field(
        description="The number of images currently waiting to be generated by this model.",
    )


# region News
class HordeNews(HordeSuccess):
    date_published: datetime.date = Field(
        description="The date this newspiece was published.",
    )
    content: str = Field(
        description="The actual piece of news.",
        alias="newspiece",
    )
    importance: str = Field(
        description="How critical this newspiece is.",
    )
# endregion


class WorkerType(Enum):
    IMAGE = "image"
    TEXT = "text"
    INTERROGATION = "interrogation"


class WorkerKudoDetails(HordeModel):
    generated: float | None = Field(
        description="The amount of kudos this worker has generated.",
    )
    uptime: int | None = Field(
        description="The amount of kudos this worker has received from staying online longer.",
    )


class TinyTeam(HordeModel):
    id: str = Field(
        description="The team's UUID.",
    )
    name: str = Field(
        description="The team's name.",
    )


class TinyWorker(HordeModel):
    id: str = Field(
        description="The worker's UUID.",
    )
    name: str = Field(
        description="The worker's name.",
    )
    online: bool = Field(
        description="Whether this worker is currently online.",
    )
    type: WorkerType = Field(
        description="The worker's type.",
    )


# noinspection SpellCheckingInspection
class Worker(TinyWorker, HordeSuccess):
    info: str | None = Field(
        default=None,
        description="The worker's info/description.",
    )
    owner: str | None = Field(
        default=None,
        description="The username (alias#id) of the user who created this worker.",
    )
    team: TinyTeam | None = Field(
        description="The team this worker belongs to.",
    )
    nsfw: bool = Field(
        description="Whether this worker will generate NSFW requests.",
    )
    models: list[str] | None = Field(
        default=None,
        description="A list of models currently served by this worker.",
    )
    forms: list[str] | None = Field(
        default=None,
        description="A list of forms currently served by this worker.",
    )
    trusted: bool = Field(
        description="Whether this worker is trusted to return valid generations.",
    )
    maintenance_mode: bool = Field(
        description="Whether this worker is currently in maintenance mode.",
    )
    flagged: bool = Field(
        description=(
            "Whether this worker has been flagged for suspicious activity. "
            "If this is true, the worker will not be given any new jobs."
        ),
    )
    bridge_agent: str = Field(
        description='The workers bridge agent name, version and website. Format: "agent name:version:website"',
        max_length=1000,
    )

    performance: str = Field(
        description="The average performance of this worker in human readable form.",
    )
    tokens_generated: float = Field(  # API docs say this is a float but that makes no sense?
        default=0,
        description="The number of tokens this worker has generated.",
    )
    megapixelsteps_generated: float = Field(
        default=0,
        description="The number of megapixelsteps this worker has generated.",
    )
    requests_fulfilled: int = Field(
        description="The number of jobs this worker has fulfilled.",
    )
    uncompleted_jobs: int = Field(
        description="The number of jobs this worker has left uncompleted after it started them.",
    )
    kudos_rewards: float = Field(
        description="The amount of kudos this worker has been rewarded in total.",
    )
    kudos_details: WorkerKudoDetails | None = Field(
        description="Details about the worker's kudos.",
    )

    # noinspection PyNestedDecorators
    @field_validator("kudos_details", mode="before")
    @classmethod
    def validate_kudos_details(cls, value: dict[str, Any]) -> WorkerKudoDetails | None:
        if not any(value.values()):
            return None
        return WorkerKudoDetails(**value)

    threads: int = Field(
        description="The number of threads this worker is currently running.",
    )
    uptime: datetime.timedelta = Field(
        description="The amount of time this worker has been online.",
    )
    max_pixels: int | None = Field(
        default=None,
        description="The maximum number of pixels this worker can generate.",
    )
    max_length: int | None = Field(
        default=None,
        description="The maximum amount of tokens this worker can generate.",
    )
    max_context_length: int | None = Field(
        default=None,
        description="The maximum amount of tokens this worker can read.",
    )

    img2img: bool = Field(
        default=False,
        description="Whether this worker allows generating images from other images.",
    )
    painting: bool = Field(
        default=False,
        description="Whether this worker allows inpainting/outpainting.",  # TODO: Factcheck, docs only list inpainting
    )
    lora: bool = Field(
        default=False,
        description="Whether this worker allows generating using LoRAs.",
    )
    post_processing: bool = Field(
        default=False,
        description="Whether this worker allows post-processing.",
        validation_alias="post-processing",
    )

    # Privileged
    suspicious: int | None = Field(
        default=None,
        description="PRIVILEGED! How much suspicion this worker has accumulated.",
    )
    paused: bool | None = Field(
        default=None,
        description="PRIVILEGED! If true, this worker will not be given any new requests.",
    )
    ipaddr: str | None = Field(
        default=None,
        description="PRIVILEGED! The last known IP address of this worker.",
    )
    contact: str | None = Field(
        default=None,
        description="PRIVILEGED! Contact details for the horde admins to reach the worker owner in case of emergency.",
    )

    # noinspection PyNestedDecorators
    @field_validator("team", mode="before")
    @classmethod
    def validate_team(cls, value: dict[str, Any]) -> TinyTeam | None:
        if not any(value.values()):
            return None
        return TinyTeam(**value)


class Team(TinyTeam, HordeSuccess):
    info: str = Field(
        description="The team's info/description.",
    )
    creator: str = Field(
        description="The username (alias#id) of the user who created this team.",
    )

    requests_fulfilled: int = Field(
        description="The number of jobs this team has fulfilled.",
    )
    kudos: float = Field(
        description="The total amount of kudos the workers in this team have generated while part of it.",
    )
    uptime: datetime.timedelta = Field(
        description="The combined amount of time workers have stayed online for while on this team.",
    )
    worker_count: int = Field(
        description="The number of workers currently in this team.",
    )
    workers: list[TinyWorker] = Field(
        description="A list of workers currently in this team.",
    )
    models: list[TinyActiveModel] = Field(
        description="A list of models currently served by this team.",
    )


# region User
class UserKudoDetails(HordeModel):
    # TODO: Rename properties to be more clear without needing to read the description
    accumulated: float = Field(
        description="The amount of kudos accumulated or used for generating images.",
    )
    recurring: float = Field(
        description="The amount of kudos this user has received from recurring rewards.",
    )
    awarded: float = Field(
        description="The amount of kudos this user has been awarded from things like rating images.",
    )

    gifted: float = Field(
        description="The amount of kudos this user has given to other users.",
    )
    received: float = Field(
        description="The amount of kudos this user has been given by other users.",
    )
    admin: float = Field(
        description="The amount of kudos this user has been given by the Horde admins.",
    )


# noinspection SpellCheckingInspection
class UserUsageDetails(HordeModel):
    megapixelsteps: float | None = Field(
        default=None,
        description="How many megapixelsteps this user has requested.",
    )
    requests: int | None = Field(
        default=None,
        description="How many images this user has requested.",
    )


# noinspection SpellCheckingInspection
class UserContributionDetails(HordeModel):
    megapixelsteps: float | None = Field(
        description="How many megapixelsteps this user has generated.",
    )
    fulfillments: int | None = Field(
        default=None,
        description="How many images this user has generated.",
    )


# noinspection SpellCheckingInspection
class UserGenerationsRecord(HordeModel):
    """Weather this object represents the user's usage or contributions depends on how this object was obtained."""

    image: int = Field(
        description="How many images this user has generated/requested.",
    )
    text: int = Field(
        description="How many texts this user has generated/requested.",
    )
    interrogation: int = Field(
        description="How many interrogations this user has generated/requested.",
    )


# noinspection SpellCheckingInspection
class UserAmountRecord(HordeModel):
    """Weather this object represents the user's usage or contributions depends on how this object was obtained."""

    megapixelsteps: float = Field(
        description="How many megapixelsteps this user has generated/requested.",
    )
    tokens: int = Field(
        description="How many token this user has generated/requested.",
    )


class UserRecords(HordeModel):
    usage: UserAmountRecord
    contribution: UserAmountRecord

    fulfillment: UserGenerationsRecord
    request: UserGenerationsRecord


class MonthlyKudos(HordeModel):
    amount: float | None = Field(
        default=None,
        description="How many kudos this user is scheduled to receive each month.",
    )
    last_received: datetime.datetime | None = Field(
        default=None,
        description="When this user last received their monthly kudo reward.",
    )


# noinspection SpellCheckingInspection
class HordeUser(HordeSuccess):
    @property
    def alias(self) -> str:
        *alias, _ = self.username.split("#")
        return "#".join(alias)

    username: str = Field(
        description=(
            "The user's unique username. "
            "It is a combination of their chosen alias plus their ID. "
            "If only the alias is wanted, use the `alias` property instead."
        ),
    )
    id: int = Field(
        description="The user's unique ID.",
    )
    account_age: datetime.timedelta = Field(
        description="How long this user has been registered.",
    )
    moderator: bool = Field(
        description="Whether this user is a moderator.",
    )
    trusted: bool = Field(
        description="Whether this user is a trusted member of the Horde.",
    )
    service: bool = Field(
        description="Whether this is a service account used by a horde proxy.",
    )
    pseudonymous: bool = Field(
        description="Whether this user has registered using an non-oauth service.",
    )
    concurrency: int = Field(
        description="How many concurrent generations this user may request.",
    )
    worker_invited: int = Field(
        description="How many workers this user is allowed to invite to the horde",
    )

    worker_count: int = Field(
        description="How many workers this user has created (active or inactive).",
    )
    worker_ids: list[str] | None = Field(
        default=None,
        description=(
            "PRIVILEGED IF NOT SET TO PUBLIC! "
            "The IDs of the workers this user has created (active or inactive)."
        ),
    )

    kudos: float = Field(
        description=(
            "The amount of kudos this user has. "
            "The amount of kudos determines the priority when requesting image generations."
        ),
    )
    kudos_details: UserKudoDetails = Field(
        description="Details about the user's kudos.",
    )
    monthly_kudos: MonthlyKudos | None = Field(
        default=None,
        description="The user's monthly kudo reward.",
    )
    usage: UserUsageDetails = Field(
        description="Details about the user's usage.",
    )
    contributions: UserContributionDetails = Field(
        description="Details about the user's contributions.",
    )
    records: UserRecords = Field(
        description="The user's records.",
    )

    # Privileged
    evaluating_kudos: float | None = Field(
        default=None,
        description=(
            "PRIVILEGED! "
            "The amount of Evaluating kudos this untrusted user has from generations and uptime. "
            "When this number reaches a pre-specified threshold, they automatically become trusted."
        ),
    )
    suspicious: int | None = Field(
        default=None,
        description="PRIVILEGED! How much suspicion this user has accumulated.",
    )
    flagged: bool | None = Field(
        default=None,
        description="PRIVILEGED! Whether this user has been flagged for suspicious activity.",
    )
    vpn: bool | None = Field(
        default=None,
        description="PRIVILEGED! Whether this user has been given the VPN role.",
    )
    special: bool | None = Field(
        default=None,
        description="PRIVILEGED! Whether this user has been given the Special role.",
    )
    contact: str | None = Field(
        default=None,
        description="PRIVILEGED! Contact details for the horde admins to reach the user in case of emergency.",
    )
    admin_comment: str | None = Field(
        default=None,
        description="PRIVILEGED! Comments left by the admins about this user.",
    )
    sharedkey_ids: list[str] | None = Field(
        default=None,
        description="PRIVILEGED! A list of shared key IDs created by this user.",
    )
# endregion


class GenerationResponse(HordeSuccess):
    id: str = Field(
        description="The UUID of the request. Use this to retrieve the request status in the future.",
    )
    kudos: int = Field(
        description="The expected kudos consumption for this request.",
    )
    message: str | None = Field(
        default=None,
        description="Any extra information from the horde about this request.",
    )


class GenerationCheck(HordeSuccess):
    waiting: int = Field(
        description="The amount of jobs waiting to be picked up by a worker.",
    )
    processing: int = Field(
        description="The amount of still processing jobs in this request.",
    )
    finished: int = Field(
        description="The amount of finished jobs in this request.",
    )
    restarted: int = Field(
        description="The amount of jobs that timed out and had to be restarted or were reported as failed by a worker.",
    )

    is_possible: bool = Field(
        description="If false, this request won't be able to be completed with the current pool of available workers.",
    )
    done: bool = Field(
        description="True when all jobs in this request are done.",
    )
    faulted: bool = Field(
        description="True when this request caused an internal server error and could not be completed.",
    )

    queue_position: int = Field(
        description="The position in the requests queue. This position is determined by relative Kudos amounts.",
    )
    wait_time: int = Field(
        description="The estimated time in seconds until all jobs in this request are done.",
    )
    kudos: float = Field(
        description="The amount of total Kudos this request has consumed until now.",
    )
