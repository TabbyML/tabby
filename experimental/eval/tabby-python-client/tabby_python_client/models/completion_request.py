from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.debug_options import DebugOptions
    from ..models.segments import Segments


T = TypeVar("T", bound="CompletionRequest")


@attr.s(auto_attribs=True)
class CompletionRequest:
    r"""
    Example:
        {'language': 'python', 'segments': {'prefix': 'def fib(n):\n    ', 'suffix': '\n        return fib(n - 1) +
            fib(n - 2)'}}

    Attributes:
        language (Union[Unset, None, str]): Language identifier, full list is maintained at
            https://code.visualstudio.com/docs/languages/identifiers Example: python.
        segments (Union[Unset, None, Segments]):
        user (Union[Unset, None, str]): A unique identifier representing your end-user, which can help Tabby to monitor
            & generating
            reports.
        debug_options (Union[Unset, None, DebugOptions]):
    """

    language: Union[Unset, None, str] = UNSET
    segments: Union[Unset, None, "Segments"] = UNSET
    user: Union[Unset, None, str] = UNSET
    debug_options: Union[Unset, None, "DebugOptions"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        language = self.language
        segments: Union[Unset, None, Dict[str, Any]] = UNSET
        if not isinstance(self.segments, Unset):
            segments = self.segments.to_dict() if self.segments else None

        user = self.user
        debug_options: Union[Unset, None, Dict[str, Any]] = UNSET
        if not isinstance(self.debug_options, Unset):
            debug_options = self.debug_options.to_dict() if self.debug_options else None

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if language is not UNSET:
            field_dict["language"] = language
        if segments is not UNSET:
            field_dict["segments"] = segments
        if user is not UNSET:
            field_dict["user"] = user
        if debug_options is not UNSET:
            field_dict["debug_options"] = debug_options

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.debug_options import DebugOptions
        from ..models.segments import Segments

        d = src_dict.copy()
        language = d.pop("language", UNSET)

        _segments = d.pop("segments", UNSET)
        segments: Union[Unset, None, Segments]
        if _segments is None:
            segments = None
        elif isinstance(_segments, Unset):
            segments = UNSET
        else:
            segments = Segments.from_dict(_segments)

        user = d.pop("user", UNSET)

        _debug_options = d.pop("debug_options", UNSET)
        debug_options: Union[Unset, None, DebugOptions]
        if _debug_options is None:
            debug_options = None
        elif isinstance(_debug_options, Unset):
            debug_options = UNSET
        else:
            debug_options = DebugOptions.from_dict(_debug_options)

        completion_request = cls(
            language=language,
            segments=segments,
            user=user,
            debug_options=debug_options,
        )

        completion_request.additional_properties = d
        return completion_request

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
