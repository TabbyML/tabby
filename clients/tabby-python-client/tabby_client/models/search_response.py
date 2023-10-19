from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.hit import Hit


T = TypeVar("T", bound="SearchResponse")


@attr.s(auto_attribs=True)
class SearchResponse:
    """
    Attributes:
        num_hits (int):
        hits (List['Hit']):
    """

    num_hits: int
    hits: List["Hit"]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        num_hits = self.num_hits
        hits = []
        for hits_item_data in self.hits:
            hits_item = hits_item_data.to_dict()

            hits.append(hits_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "num_hits": num_hits,
                "hits": hits,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.hit import Hit

        d = src_dict.copy()
        num_hits = d.pop("num_hits")

        hits = []
        _hits = d.pop("hits")
        for hits_item_data in _hits:
            hits_item = Hit.from_dict(hits_item_data)

            hits.append(hits_item)

        search_response = cls(
            num_hits=num_hits,
            hits=hits,
        )

        search_response.additional_properties = d
        return search_response

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
