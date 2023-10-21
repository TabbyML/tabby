from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="Version")


@attr.s(auto_attribs=True)
class Version:
    """
    Attributes:
        build_date (str):
        build_timestamp (str):
        git_sha (str):
        git_describe (str):
    """

    build_date: str
    build_timestamp: str
    git_sha: str
    git_describe: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        build_date = self.build_date
        build_timestamp = self.build_timestamp
        git_sha = self.git_sha
        git_describe = self.git_describe

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "build_date": build_date,
                "build_timestamp": build_timestamp,
                "git_sha": git_sha,
                "git_describe": git_describe,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        build_date = d.pop("build_date")

        build_timestamp = d.pop("build_timestamp")

        git_sha = d.pop("git_sha")

        git_describe = d.pop("git_describe")

        version = cls(
            build_date=build_date,
            build_timestamp=build_timestamp,
            git_sha=git_sha,
            git_describe=git_describe,
        )

        version.additional_properties = d
        return version

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
