"""Deterministic runtime generation, utterance, and playback lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from uuid import uuid4


class GenerationState(str, Enum):
    ACTIVE = "active"
    SEALED = "sealed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class UtteranceState(str, Enum):
    SYNTHESIZING = "synthesizing"
    READY = "ready"
    PLAYING = "playing"
    PLAYED = "played"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class UtteranceRecord:
    utterance_id: str
    generation_id: str
    text: str
    include_in_history: bool = True
    state: UtteranceState = UtteranceState.SYNTHESIZING
    synthesis_finished: bool = False
    chunk_ids: list[str] = field(default_factory=list)
    played_chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerationRecord:
    generation_id: str
    turn_id: str
    revision: int
    state: GenerationState = GenerationState.ACTIVE
    utterance_ids: list[str] = field(default_factory=list)


class OutputController:
    """The sole authority for admitting generated audio into playback."""

    def __init__(self) -> None:
        self.current_generation_id: str | None = None
        self._generations: dict[str, GenerationRecord] = {}
        self._utterances: dict[str, UtteranceRecord] = {}

    def begin_generation(
        self, turn_id: str, revision: int, *, generation_id: str | None = None,
        preserve_playback: bool = False,
    ) -> GenerationRecord:
        for live_id in self.live_generation_ids():
            current = self._generations[live_id]
            if preserve_playback:
                if current.state is GenerationState.ACTIVE:
                    raise ValueError("finish active decoding before queuing another generation")
            else:
                self.cancel_generation(live_id)
        record = GenerationRecord(
            generation_id=generation_id or str(uuid4()),
            turn_id=turn_id,
            revision=revision,
        )
        if record.generation_id in self._generations:
            raise ValueError(f"generation already exists: {record.generation_id}")
        self._generations[record.generation_id] = record
        self.current_generation_id = record.generation_id
        return replace(record, utterance_ids=list(record.utterance_ids))

    def begin_utterance(
        self,
        generation_id: str,
        text: str,
        *,
        utterance_id: str | None = None,
        include_in_history: bool = True,
    ) -> UtteranceRecord:
        generation = self._require_current_generation(generation_id)
        if generation.state is not GenerationState.ACTIVE:
            raise ValueError("cannot add an utterance after generation is sealed")
        record = UtteranceRecord(
            utterance_id=utterance_id or str(uuid4()),
            generation_id=generation_id,
            text=text,
            include_in_history=include_in_history,
        )
        if record.utterance_id in self._utterances:
            raise ValueError(f"utterance already exists: {record.utterance_id}")
        self._utterances[record.utterance_id] = record
        generation.utterance_ids.append(record.utterance_id)
        return self._copy_utterance(record)

    def admit_audio_chunk(
        self, generation_id: str, utterance_id: str, chunk_id: str
    ) -> bool:
        if not chunk_id:
            return False
        generation = self._generations.get(generation_id)
        utterance = self._utterances.get(utterance_id)
        if generation is None or utterance is None:
            return False
        if generation.state in (GenerationState.CANCELLED, GenerationState.COMPLETED):
            return False
        if utterance.generation_id != generation_id or utterance.state in (
            UtteranceState.CANCELLED,
            UtteranceState.PLAYED,
        ):
            return False
        if chunk_id in utterance.chunk_ids:
            return False
        utterance.chunk_ids.append(chunk_id)
        return True

    def append_utterance_text(
        self, generation_id: str, utterance_id: str, text: str
    ) -> bool:
        """Append an accepted model text delta before synthesis is finalized."""

        utterance = self._get_live_utterance(generation_id, utterance_id)
        if utterance is None or utterance.synthesis_finished or not text:
            return False
        utterance.text += text
        return True

    def revoke_audio_chunk(
        self, generation_id: str, utterance_id: str, chunk_id: str
    ) -> bool:
        """Undo the newest admission when transport capture fails."""
        utterance = self._get_live_utterance(generation_id, utterance_id)
        if utterance is None or not utterance.chunk_ids:
            return False
        if (
            utterance.chunk_ids[-1] != chunk_id
            or chunk_id in utterance.played_chunk_ids
        ):
            return False
        utterance.chunk_ids.pop()
        return True

    def finish_synthesis(self, generation_id: str, utterance_id: str) -> bool:
        utterance = self._get_live_utterance(generation_id, utterance_id)
        if utterance is None or utterance.synthesis_finished:
            return False
        utterance.synthesis_finished = True
        if utterance.state is UtteranceState.SYNTHESIZING:
            utterance.state = UtteranceState.READY
        return True

    def mark_playing(
        self,
        generation_id: str,
        utterance_id: str,
        chunk_id: str | None = None,
    ) -> bool:
        utterance = self._get_live_utterance(generation_id, utterance_id)
        if utterance is None or not utterance.chunk_ids:
            return False
        if chunk_id is not None and chunk_id not in utterance.chunk_ids:
            return False
        utterance.state = UtteranceState.PLAYING
        return True

    def confirm_chunk_played(
        self, generation_id: str, utterance_id: str, chunk_id: str
    ) -> bool:
        utterance = self._get_live_utterance(generation_id, utterance_id)
        if utterance is None or chunk_id not in utterance.chunk_ids:
            return False
        if chunk_id in utterance.played_chunk_ids:
            return False
        next_index = len(utterance.played_chunk_ids)
        if utterance.chunk_ids[next_index] != chunk_id:
            return False
        utterance.played_chunk_ids.append(chunk_id)
        utterance.state = UtteranceState.PLAYING
        return True

    def confirm_playback_finished(
        self, generation_id: str, utterance_id: str
    ) -> bool:
        utterance = self._get_live_utterance(generation_id, utterance_id)
        if (
            utterance is None
            or not utterance.synthesis_finished
            or utterance.played_chunk_ids != utterance.chunk_ids
        ):
            return False
        utterance.state = UtteranceState.PLAYED
        generation = self._generations[generation_id]
        if generation.state is GenerationState.SEALED and all(
            self._utterances[item].state is UtteranceState.PLAYED
            for item in generation.utterance_ids
        ):
            generation.state = GenerationState.COMPLETED
        return True

    def seal_generation(self, generation_id: str) -> bool:
        generation = self._generations.get(generation_id)
        if generation is None or generation_id != self.current_generation_id:
            return False
        if generation.state is not GenerationState.ACTIVE:
            return False
        generation.state = GenerationState.SEALED
        if not generation.utterance_ids or all(
            self._utterances[item].state is UtteranceState.PLAYED
            for item in generation.utterance_ids
        ):
            generation.state = GenerationState.COMPLETED
        return True

    def cancel_generation(self, generation_id: str) -> bool:
        generation = self._generations.get(generation_id)
        if generation is None or generation.state in (
            GenerationState.CANCELLED,
            GenerationState.COMPLETED,
        ):
            return False
        generation.state = GenerationState.CANCELLED
        for utterance_id in generation.utterance_ids:
            utterance = self._utterances[utterance_id]
            if utterance.state is not UtteranceState.PLAYED:
                utterance.state = UtteranceState.CANCELLED
        return True

    def generation(self, generation_id: str) -> GenerationRecord | None:
        record = self._generations.get(generation_id)
        if record is None:
            return None
        return replace(record, utterance_ids=list(record.utterance_ids))

    def is_generation_live(self, generation_id: str) -> bool:
        generation = self._generations.get(generation_id)
        return (
            generation is not None
            and generation.state in (GenerationState.ACTIVE, GenerationState.SEALED)
        )

    def utterance(self, utterance_id: str) -> UtteranceRecord | None:
        record = self._utterances.get(utterance_id)
        return None if record is None else self._copy_utterance(record)

    def _require_current_generation(self, generation_id: str) -> GenerationRecord:
        if generation_id != self.current_generation_id:
            raise ValueError(f"generation is stale: {generation_id}")
        generation = self._generations.get(generation_id)
        if generation is None or generation.state is GenerationState.CANCELLED:
            raise ValueError(f"generation is not active: {generation_id}")
        return generation

    def _get_live_utterance(
        self, generation_id: str, utterance_id: str
    ) -> UtteranceRecord | None:
        generation = self._generations.get(generation_id)
        utterance = self._utterances.get(utterance_id)
        if generation is None or utterance is None:
            return None
        if generation.state in (GenerationState.CANCELLED, GenerationState.COMPLETED):
            return None
        if utterance.generation_id != generation_id or utterance.state in (
            UtteranceState.CANCELLED,
            UtteranceState.PLAYED,
        ):
            return None
        return utterance

    def live_generation_ids(self) -> list[str]:
        return [key for key in self._generations if self.is_generation_live(key)]

    @staticmethod
    def _copy_utterance(record: UtteranceRecord) -> UtteranceRecord:
        return replace(
            record,
            chunk_ids=list(record.chunk_ids),
            played_chunk_ids=list(record.played_chunk_ids),
        )
