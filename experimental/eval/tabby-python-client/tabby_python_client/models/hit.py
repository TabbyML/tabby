from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.hit_document import HitDocument


T = TypeVar("T", bound="Hit")


@attr.s(auto_attribs=True)
class Hit:
    """
    Attributes:
        score (float):
        doc (HitDocument):
        id (int):
    """

    score: float
    doc: "HitDocument"
    id: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        score = self.score
        doc = self.doc.to_dict()

        id = self.id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "score": score,
                "doc": doc,
                "id": id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.hit_document import HitDocument

        d = src_dict.copy()
        score = d.pop("score")

        doc = HitDocument.from_dict(d.pop("doc"))

        id = d.pop("id")

        hit = cls(
            score=score,
            doc=doc,
            id=id,
        )

        hit.additional_properties = d
        return hit

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
