from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.message import Message


T = TypeVar("T", bound="ChatCompletionRequest")


@attr.s(auto_attribs=True)
class ChatCompletionRequest:
    """
    Example:
        {'messages': [{'content': 'What is tail recursion?', 'role': 'user'}, {'content': "It's a kind of optimization
            in compiler?", 'role': 'assistant'}, {'content': 'Could you share more details?', 'role': 'user'}]}

    Attributes:
        messages (List['Message']):
    """

    messages: List["Message"]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        messages = []
        for messages_item_data in self.messages:
            messages_item = messages_item_data.to_dict()

            messages.append(messages_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "messages": messages,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.message import Message

        d = src_dict.copy()
        messages = []
        _messages = d.pop("messages")
        for messages_item_data in _messages:
            messages_item = Message.from_dict(messages_item_data)

            messages.append(messages_item)

        chat_completion_request = cls(
            messages=messages,
        )

        chat_completion_request.additional_properties = d
        return chat_completion_request

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
