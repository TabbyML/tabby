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
  errorHandler?: (response: Response) => any
}

export default async function authEnhancedFetch(
  url: string,
  options?: FetcherOptions
): Promise<any> {
  if (willAuthError(url)) {
    return tokenManager.refreshToken(doRefreshToken).then(res => {
      return requestWithAuth(url, options)
    })
  }

  const response: Response = await fetch(url, addAuthToRequest(options))

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

  if (typeof window !== 'undefined') {
    const accessToken = getAuthToken()?.accessToken
    const fetcherOptions = getFetcherOptions()

    if (accessToken) {
      headers.append('Authorization', `Bearer ${accessToken}`)
    } else if (fetcherOptions) {
      headers.append('Authorization', `Bearer ${fetcherOptions.authorization}`)
      if (fetcherOptions.headers) {
        for (const [k, v] of Object.entries(fetcherOptions.headers)) {
          headers.append(k, v as any)
        }
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
  return fetch(url, addAuthToRequest(options)).then(x => {
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
