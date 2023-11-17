from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="DebugOptions")


@attr.s(auto_attribs=True)
class DebugOptions:
    """
    Attributes:
        raw_prompt (Union[Unset, None, str]): When `raw_prompt` is specified, it will be passed directly to the
            inference engine for completion. `segments` field in `CompletionRequest` will be ignored.

            This is useful for certain requests that aim to test the tabby's e2e quality.
        return_snippets (Union[Unset, bool]): When true, returns `snippets` in `debug_data`.
        return_prompt (Union[Unset, bool]): When true, returns `prompt` in `debug_data`.
        disable_retrieval_augmented_code_completion (Union[Unset, bool]): When true, disable retrieval augmented code
            completion.
    """

    raw_prompt: Union[Unset, None, str] = UNSET
    return_snippets: Union[Unset, bool] = UNSET
    return_prompt: Union[Unset, bool] = UNSET
    disable_retrieval_augmented_code_completion: Union[Unset, bool] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        raw_prompt = self.raw_prompt
        return_snippets = self.return_snippets
        return_prompt = self.return_prompt
        disable_retrieval_augmented_code_completion = self.disable_retrieval_augmented_code_completion

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if raw_prompt is not UNSET:
            field_dict["raw_prompt"] = raw_prompt
        if return_snippets is not UNSET:
            field_dict["return_snippets"] = return_snippets
        if return_prompt is not UNSET:
            field_dict["return_prompt"] = return_prompt
        if disable_retrieval_augmented_code_completion is not UNSET:
            field_dict["disable_retrieval_augmented_code_completion"] = disable_retrieval_augmented_code_completion

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        raw_prompt = d.pop("raw_prompt", UNSET)

        return_snippets = d.pop("return_snippets", UNSET)

        return_prompt = d.pop("return_prompt", UNSET)

        disable_retrieval_augmented_code_completion = d.pop("disable_retrieval_augmented_code_completion", UNSET)

        debug_options = cls(
            raw_prompt=raw_prompt,
            return_snippets=return_snippets,
            return_prompt=return_prompt,
            disable_retrieval_augmented_code_completion=disable_retrieval_augmented_code_completion,
        )

        debug_options.additional_properties = d
        return debug_options

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
