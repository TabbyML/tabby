from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.choice import Choice


T = TypeVar("T", bound="CompletionResponse")


@attr.s(auto_attribs=True)
class CompletionResponse:
    """
    Attributes:
        id (str):
        choices (List['Choice']):
    """

    id: str
    choices: List["Choice"]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        choices = []
        for choices_item_data in self.choices:
            choices_item = choices_item_data.to_dict()

            choices.append(choices_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "choices": choices,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.choice import Choice

        d = src_dict.copy()
        id = d.pop("id")

        choices = []
        _choices = d.pop("choices")
        for choices_item_data in _choices:
            choices_item = Choice.from_dict(choices_item_data)

            choices.append(choices_item)

        completion_response = cls(
            id=id,
            choices=choices,
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
