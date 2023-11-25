from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.version import Version


T = TypeVar("T", bound="HealthState")


@attr.s(auto_attribs=True)
class HealthState:
    """
    Attributes:
        model (str):
        device (str):
        arch (str):
        cpu_info (str):
        cpu_count (int):
        cuda_devices (List[str]):
        version (Version):
        chat_model (Union[Unset, None, str]):
    """

    model: str
    device: str
    arch: str
    cpu_info: str
    cpu_count: int
    cuda_devices: List[str]
    version: "Version"
    chat_model: Union[Unset, None, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        model = self.model
        device = self.device
        arch = self.arch
        cpu_info = self.cpu_info
        cpu_count = self.cpu_count
        cuda_devices = self.cuda_devices

        version = self.version.to_dict()

        chat_model = self.chat_model

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "model": model,
                "device": device,
                "arch": arch,
                "cpu_info": cpu_info,
                "cpu_count": cpu_count,
                "cuda_devices": cuda_devices,
                "version": version,
            }
        )
        if chat_model is not UNSET:
            field_dict["chat_model"] = chat_model

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.version import Version

        d = src_dict.copy()
        model = d.pop("model")

        device = d.pop("device")

        arch = d.pop("arch")

        cpu_info = d.pop("cpu_info")

        cpu_count = d.pop("cpu_count")

        cuda_devices = cast(List[str], d.pop("cuda_devices"))

        version = Version.from_dict(d.pop("version"))

        chat_model = d.pop("chat_model", UNSET)

        health_state = cls(
            model=model,
            device=device,
            arch=arch,
            cpu_info=cpu_info,
            cpu_count=cpu_count,
            cuda_devices=cuda_devices,
            version=version,
            chat_model=chat_model,
        )

        health_state.additional_properties = d
        return health_state

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
