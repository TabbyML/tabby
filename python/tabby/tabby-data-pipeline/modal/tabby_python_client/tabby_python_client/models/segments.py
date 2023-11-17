from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="Segments")


@attr.s(auto_attribs=True)
class Segments:
    """
    Attributes:
        prefix (str): Content that appears before the cursor in the editor window.
        suffix (Union[Unset, None, str]): Content that appears after the cursor in the editor window.
    """

    prefix: str
    suffix: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        prefix = self.prefix
        suffix = self.suffix

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "prefix": prefix,
            }
        )
        if suffix is not UNSET:
            field_dict["suffix"] = suffix

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        prefix = d.pop("prefix")

        suffix = d.pop("suffix", UNSET)

        segments = cls(
            prefix=prefix,
            suffix=suffix,
        )

        segments.additional_properties = d
        return segments

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
