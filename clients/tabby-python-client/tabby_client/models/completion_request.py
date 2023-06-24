from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.segments import Segments


T = TypeVar("T", bound="CompletionRequest")


@attr.s(auto_attribs=True)
class CompletionRequest:
    r"""
    Example:
        {'language': 'python', 'segments': {'prefix': 'def fib(n):\n    ', 'suffix': '\n        return fib(n - 1) +
            fib(n - 2)'}}

    Attributes:
        prompt (Union[Unset, None, str]):  Example: def fib(n):.
        language (Union[Unset, None, str]): Language identifier, full list is maintained at
            https://code.visualstudio.com/docs/languages/identifiers Example: python.
        segments (Union[Unset, None, Segments]):
        user (Union[Unset, None, str]):
    """

    prompt: Union[Unset, None, str] = UNSET
    language: Union[Unset, None, str] = UNSET
    segments: Union[Unset, None, "Segments"] = UNSET
    user: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        prompt = self.prompt
        language = self.language
        segments: Union[Unset, None, Dict[str, Any]] = UNSET
        if not isinstance(self.segments, Unset):
            segments = self.segments.to_dict() if self.segments else None

        user = self.user

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if prompt is not UNSET:
            field_dict["prompt"] = prompt
        if language is not UNSET:
            field_dict["language"] = language
        if segments is not UNSET:
            field_dict["segments"] = segments
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.segments import Segments

        d = src_dict.copy()
        prompt = d.pop("prompt", UNSET)

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

        completion_request = cls(
            prompt=prompt,
            language=language,
            segments=segments,
            user=user,
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
