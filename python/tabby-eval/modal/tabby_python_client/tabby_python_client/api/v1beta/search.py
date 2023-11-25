from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.search_response import SearchResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    q: str = "get",
    limit: Union[Unset, None, int] = 20,
    offset: Union[Unset, None, int] = 0,
) -> Dict[str, Any]:
    url = "{}/v1beta/search".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["q"] = q

    params["limit"] = limit

    params["offset"] = offset

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[Any, SearchResponse]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = SearchResponse.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.NOT_IMPLEMENTED:
        response_501 = cast(Any, None)
        return response_501
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[Any, SearchResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    q: str = "get",
    limit: Union[Unset, None, int] = 20,
    offset: Union[Unset, None, int] = 0,
) -> Response[Union[Any, SearchResponse]]:
    """
    Args:
        q (str):  Default: 'get'.
        limit (Union[Unset, None, int]):  Default: 20.
        offset (Union[Unset, None, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, SearchResponse]]
    """

    kwargs = _get_kwargs(
        client=client,
        q=q,
        limit=limit,
        offset=offset,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    q: str = "get",
    limit: Union[Unset, None, int] = 20,
    offset: Union[Unset, None, int] = 0,
) -> Optional[Union[Any, SearchResponse]]:
    """
    Args:
        q (str):  Default: 'get'.
        limit (Union[Unset, None, int]):  Default: 20.
        offset (Union[Unset, None, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, SearchResponse]
    """

    return sync_detailed(
        client=client,
        q=q,
        limit=limit,
        offset=offset,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    q: str = "get",
    limit: Union[Unset, None, int] = 20,
    offset: Union[Unset, None, int] = 0,
) -> Response[Union[Any, SearchResponse]]:
    """
    Args:
        q (str):  Default: 'get'.
        limit (Union[Unset, None, int]):  Default: 20.
        offset (Union[Unset, None, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, SearchResponse]]
    """

    kwargs = _get_kwargs(
        client=client,
        q=q,
        limit=limit,
        offset=offset,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    q: str = "get",
    limit: Union[Unset, None, int] = 20,
    offset: Union[Unset, None, int] = 0,
) -> Optional[Union[Any, SearchResponse]]:
    """
    Args:
        q (str):  Default: 'get'.
        limit (Union[Unset, None, int]):  Default: 20.
        offset (Union[Unset, None, int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, SearchResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            q=q,
            limit=limit,
            offset=offset,
        )
    ).parsed
