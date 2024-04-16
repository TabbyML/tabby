import { createRequest } from '@urql/core'

import {
  clearAuthToken,
  getAuthToken,
  refreshTokenMutation,
  saveAuthToken
} from './auth'
import { client } from './gql'

interface FetcherOptions extends RequestInit {
  responseFormat?: 'json' | 'blob'
  responseFormatter?: (response: Response) => any
  customFetch?: (
    input: RequestInfo | URL,
    init?: RequestInit | undefined
  ) => Promise<Response>,
  errorHandler?: (response: Response) => any
}
interface PendingRequest {
  url: string
  options?: FetcherOptions
  resolve: Function
}
let refreshing = false
const queue: PendingRequest[] = []

export default async function authEnhancedFetch(
  url: string,
  options?: FetcherOptions
): Promise<any> {
  const currentFetcher = options?.customFetch ?? window.fetch
  const response: Response = await currentFetcher(
    url,
    addAuthToRequest(options)
  )
  
  if (response.status === 401) {
    if (refreshing) {
      return new Promise(resolve => {
        queue.push({ url, options, resolve })
      })
    }

    const refreshToken = getAuthToken()?.refreshToken
    if (!refreshToken) {
      clearAuthToken()
      return
    }

    refreshing = true

    const refreshAuthRes = await refreshAuth(refreshToken)

    const newToken = refreshAuthRes?.data?.refreshToken
    if (newToken) {
      saveAuthToken({
        accessToken: newToken.accessToken,
        refreshToken: newToken.refreshToken
      })
      refreshing = false
      while (queue.length) {
        const q = queue.shift()
        q?.resolve(requestWithAuth(q.url, q.options))
      }

      return requestWithAuth(url, options)
    } else {
      refreshing = false
      queue.length = 0
      clearAuthToken()
    }
  } else {
    return formatResponse(response, options)
  }
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
  // refreshAuth
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
  options?: Pick<FetcherOptions, 'responseFormat' | 'responseFormatter' | 'errorHandler'>
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
