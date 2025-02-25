import { Client, createRequest, fetchExchange } from '@urql/core'
import { jwtDecode } from 'jwt-decode'

import { refreshTokenMutation } from './auth'
import {
  getAuthToken,
  getFetcherOptions,
  isTokenExpired,
  tokenManager
} from './token-management'

interface FetcherOptions extends RequestInit {
  responseFormat?: 'json' | 'blob'
  responseFormatter?: (response: Response) => any
  customFetch?: (
    input: RequestInfo | URL,
    init?: RequestInit | undefined
  ) => Promise<Response>
  errorHandler?: (response: Response) => any
}

export default async function authEnhancedFetch(
  url: string,
  options?: FetcherOptions
): Promise<any> {
  const currentFetcher = options?.customFetch ?? window.fetch
  if (willAuthError(url)) {
    return tokenManager.refreshToken(doRefreshToken).then(res => {
      return requestWithAuth(url, options)
    })
  }

  const response: Response = await currentFetcher(
    url,
    addAuthToRequest(options)
  )

  if (response.status === 401) {
    tokenManager.clearAccessToken()

    return tokenManager.refreshToken(doRefreshToken).then(res => {
      return requestWithAuth(url, options)
    })
  } else {
    return formatResponse(response, options)
  }
}

function willAuthError(url: string) {
  if (url.startsWith('/oauth/providers')) {
    return false
  }
  const accessToken = getAuthToken()?.accessToken
  const fetcherOptions = getFetcherOptions()
  if (accessToken) {
    // Check whether `token` JWT is expired
    try {
      const { exp } = jwtDecode(accessToken)
      return isTokenExpired(exp)
    } catch (e) {
      return true
    }
  } else if (fetcherOptions) {
    return !fetcherOptions?.authorization
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
  const accessToken = getAuthToken()?.accessToken
  const fetcherOptions = getFetcherOptions()
  if (typeof window !== 'undefined') {
    if (accessToken) {
      headers.append('authorization', `Bearer ${getAuthToken()?.accessToken}`)
    } else if (fetcherOptions) {
      const newHeaders: Record<string, any> = {
        Authorization: `Bearer ${fetcherOptions.authorization}`,
        ...fetcherOptions.headers
      }
      const keys = Object.keys(newHeaders)
      for (const key of keys) {
        headers.append(key, newHeaders[key])
      }
    }
  }

  return {
    ...(options || {}),
    headers
  }
}

async function refreshAuth(refreshToken: string) {
  const client = new Client({
    url: `/graphql`,
    requestPolicy: 'network-only',
    exchanges: [fetchExchange]
  })
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
  options?: Pick<
    FetcherOptions,
    'responseFormat' | 'responseFormatter' | 'errorHandler'
  >
) {
  if (!response?.ok) {
    if (options?.errorHandler) return options.errorHandler(response)
    return undefined
  }
  if (options?.responseFormatter) {
    return options.responseFormatter(response)
  }

  if (options?.responseFormat === 'blob') {
    return response.blob()
  }

  return response.json()
}
