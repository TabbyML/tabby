import { createRequest } from '@urql/core'
import { jwtDecode } from 'jwt-decode'

import { refreshTokenMutation } from './auth'
import { client } from './gql'
import {
  getAuthToken,
  isTokenExpired,
  tokenManagerInstance
} from './token-management'

interface FetcherOptions extends RequestInit {
  responseFormat?: 'json' | 'blob'
  responseFormatter?: (response: Response) => any
  customFetch?: (
    input: RequestInfo | URL,
    init?: RequestInit | undefined
  ) => Promise<Response>
}

export default async function authEnhancedFetch(
  url: string,
  options?: FetcherOptions
): Promise<any> {
  const currentFetcher = options?.customFetch ?? window.fetch

  if (willAuthError()) {
    return tokenManagerInstance.refreshToken(doRefreshToken).then(res => {
      return requestWithAuth(url, options)
    })
  }

  const response: Response = await currentFetcher(
    url,
    addAuthToRequest(options)
  )

  if (response.status === 401) {
    return tokenManagerInstance.refreshToken(doRefreshToken).then(res => {
      return requestWithAuth(url, options)
    })
  } else {
    return formatResponse(response, options)
  }
}

function willAuthError() {
  const accessToken = getAuthToken()?.accessToken
  if (accessToken) {
    // Check whether `token` JWT is expired
    try {
      const { exp } = jwtDecode(accessToken)
      return exp ? isTokenExpired(exp) : true
    } catch (e) {
      return true
    }
  } else {
    return true
  }
}

async function doRefreshToken() {
  let refreshToken = getAuthToken()?.refreshToken
  if (!refreshToken) return undefined

  const newToken = await refreshAuth(refreshToken)
  return newToken?.data?.refreshToken
}

function addAuthToRequest(options?: FetcherOptions): FetcherOptions {
  const headers = new Headers(options?.headers)

  if (typeof window !== 'undefined') {
    headers.append('authorization', `Bearer ${getAuthToken()?.accessToken}`)
  }

  return {
    ...(options || {}),
    headers
  }
}

async function refreshAuth(refreshToken: string) {
  const refreshAuth = client.createRequestOperation(
    'mutation',
    createRequest(refreshTokenMutation, { refreshToken })
  )
  return client.executeMutation(refreshAuth)
}

function requestWithAuth(url: string, options?: FetcherOptions) {
  const currentFetcher = options?.customFetch ?? window.fetch
  return currentFetcher(url, addAuthToRequest(options)).then(x => {
    return formatResponse(x, options)
  })
}

function formatResponse(
  response: Response,
  options?: Pick<FetcherOptions, 'responseFormat' | 'responseFormatter'>
) {
  if (!response?.ok) return undefined
  if (options?.responseFormatter) {
    return options.responseFormatter(response)
  }

  if (options?.responseFormat === 'blob') {
    return response.blob()
  }

  return response.json()
}
