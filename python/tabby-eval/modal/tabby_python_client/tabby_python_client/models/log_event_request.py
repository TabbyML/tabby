from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="LogEventRequest")


@attr.s(auto_attribs=True)
class LogEventRequest:
    """
    Attributes:
        type (str): Event type, should be `view` or `select`. Example: view.
        completion_id (str):
        choice_index (int):
    """

    type: str
    completion_id: str
    choice_index: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        type = self.type
        completion_id = self.completion_id
        choice_index = self.choice_index

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type,
                "completion_id": completion_id,
                "choice_index": choice_index,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        type = d.pop("type")

        completion_id = d.pop("completion_id")

        choice_index = d.pop("choice_index")

        log_event_request = cls(
            type=type,
            completion_id=completion_id,
            choice_index=choice_index,
        )

        log_event_request.additional_properties = d
        return log_event_request

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
