from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.choice import Choice
    from ..models.debug_data import DebugData


T = TypeVar("T", bound="CompletionResponse")


@attr.s(auto_attribs=True)
class CompletionResponse:
    """
    Example:
        {'choices': [{'index': 0, 'text': 'string'}], 'id': 'string'}

    Attributes:
        id (str):
        choices (List['Choice']):
        debug_data (Union[Unset, None, DebugData]):
    """

    id: str
    choices: List["Choice"]
    debug_data: Union[Unset, None, "DebugData"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        choices = []
        for choices_item_data in self.choices:
            choices_item = choices_item_data.to_dict()

            choices.append(choices_item)

        debug_data: Union[Unset, None, Dict[str, Any]] = UNSET
        if not isinstance(self.debug_data, Unset):
            debug_data = self.debug_data.to_dict() if self.debug_data else None

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "choices": choices,
            }
        )
        if debug_data is not UNSET:
            field_dict["debug_data"] = debug_data

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.choice import Choice
        from ..models.debug_data import DebugData

        d = src_dict.copy()
        id = d.pop("id")

        choices = []
        _choices = d.pop("choices")
        for choices_item_data in _choices:
            choices_item = Choice.from_dict(choices_item_data)

            choices.append(choices_item)

        _debug_data = d.pop("debug_data", UNSET)
        debug_data: Union[Unset, None, DebugData]
        if _debug_data is None:
            debug_data = None
        elif isinstance(_debug_data, Unset):
            debug_data = UNSET
        else:
            debug_data = DebugData.from_dict(_debug_data)

        completion_response = cls(
            id=id,
            choices=choices,
            debug_data=debug_data,
        )

        completion_response.additional_properties = d
        return completion_response

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
