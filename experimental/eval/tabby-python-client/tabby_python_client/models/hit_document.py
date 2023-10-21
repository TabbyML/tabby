from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="HitDocument")


@attr.s(auto_attribs=True)
class HitDocument:
    """
    Attributes:
        body (str):
        filepath (str):
        git_url (str):
        kind (str):
        language (str):
        name (str):
    """

    body: str
    filepath: str
    git_url: str
    kind: str
    language: str
    name: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        body = self.body
        filepath = self.filepath
        git_url = self.git_url
        kind = self.kind
        language = self.language
        name = self.name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "body": body,
                "filepath": filepath,
                "git_url": git_url,
                "kind": kind,
                "language": language,
                "name": name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        body = d.pop("body")

        filepath = d.pop("filepath")

        git_url = d.pop("git_url")

        kind = d.pop("kind")

        language = d.pop("language")

        name = d.pop("name")

        hit_document = cls(
            body=body,
            filepath=filepath,
            git_url=git_url,
            kind=kind,
            language=language,
            name=name,
        )

        hit_document.additional_properties = d
        return hit_document

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
