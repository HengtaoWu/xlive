"""Internal X-Talk events carrying accepted X-Live speech inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..runtime.protocol_events import BaseEvent

from ..runtime.events import EventEnvelope


@dataclass
class XLiveUserSnapshotObserved(BaseEvent):
    TYPE: ClassVar[str] = "xlive.speech.user_snapshot"
    envelope: EventEnvelope


@dataclass
class XLiveTurnStateObserved(BaseEvent):
    TYPE: ClassVar[str] = "xlive.speech.turn_state"
    envelope: EventEnvelope


@dataclass
class XLiveUserVocalizationDiscarded(BaseEvent):
    """A VAD segment classified as non-turn speech such as a backchannel."""

    TYPE: ClassVar[str] = "xlive.speech.user_vocalization_discarded"


@dataclass
class XLiveUserCommitted(BaseEvent):
    TYPE: ClassVar[str] = "xlive.speech.user_commit"
    envelope: EventEnvelope
