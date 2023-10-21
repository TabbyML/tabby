from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.snippet import Snippet


T = TypeVar("T", bound="DebugData")


@attr.s(auto_attribs=True)
class DebugData:
    """
    Attributes:
        snippets (Union[Unset, None, List['Snippet']]):
        prompt (Union[Unset, None, str]):
    """

    snippets: Union[Unset, None, List["Snippet"]] = UNSET
    prompt: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        snippets: Union[Unset, None, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.snippets, Unset):
            if self.snippets is None:
                snippets = None
            else:
                snippets = []
                for snippets_item_data in self.snippets:
                    snippets_item = snippets_item_data.to_dict()

                    snippets.append(snippets_item)

        prompt = self.prompt

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if snippets is not UNSET:
            field_dict["snippets"] = snippets
        if prompt is not UNSET:
            field_dict["prompt"] = prompt

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.snippet import Snippet

        d = src_dict.copy()
        snippets = []
        _snippets = d.pop("snippets", UNSET)
        for snippets_item_data in _snippets or []:
            snippets_item = Snippet.from_dict(snippets_item_data)

            snippets.append(snippets_item)

        prompt = d.pop("prompt", UNSET)

        debug_data = cls(
            snippets=snippets,
            prompt=prompt,
        )

        debug_data.additional_properties = d
        return debug_data

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
