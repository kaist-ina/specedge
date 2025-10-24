from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ValidateRequest(_message.Message):
    __slots__ = (
        "client_idx",
        "req_idx",
        "input_ids",
        "position_ids",
        "cache_seq_indices",
        "parent_indices",
        "attention_mask",
        "prefill",
        "prefix",
    )
    CLIENT_IDX_FIELD_NUMBER: _ClassVar[int]
    REQ_IDX_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    POSITION_IDS_FIELD_NUMBER: _ClassVar[int]
    CACHE_SEQ_INDICES_FIELD_NUMBER: _ClassVar[int]
    PARENT_INDICES_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_MASK_FIELD_NUMBER: _ClassVar[int]
    PREFILL_FIELD_NUMBER: _ClassVar[int]
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    client_idx: int
    req_idx: int
    input_ids: bytes
    position_ids: bytes
    cache_seq_indices: bytes
    parent_indices: bytes
    attention_mask: bytes
    prefill: bool
    prefix: str
    def __init__(
        self,
        client_idx: _Optional[int] = ...,
        req_idx: _Optional[int] = ...,
        input_ids: _Optional[bytes] = ...,
        position_ids: _Optional[bytes] = ...,
        cache_seq_indices: _Optional[bytes] = ...,
        parent_indices: _Optional[bytes] = ...,
        attention_mask: _Optional[bytes] = ...,
        prefill: bool = ...,
        prefix: _Optional[str] = ...,
    ) -> None: ...

class ValidateResponse(_message.Message):
    __slots__ = ("selection", "prefill")
    SELECTION_FIELD_NUMBER: _ClassVar[int]
    PREFILL_FIELD_NUMBER: _ClassVar[int]
    selection: bytes
    prefill: int
    def __init__(
        self, selection: _Optional[bytes] = ..., prefill: _Optional[int] = ...
    ) -> None: ...

class SyncRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SyncResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
