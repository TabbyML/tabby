import { createRequest } from '@urql/core'

import {
  clearAuthToken,
  getAuthToken,
  refreshTokenMutation,
  saveAuthToken
} from './auth'
import { client } from './gql'

interface PendingRequest {
  url: string
  init?: RequestInit & { format?: 'json' | 'text' }
  resolve: Function
}
let refreshing = false
const queue: PendingRequest[] = []

export default async function tokenFetcher(
  url: string,
  init?: PendingRequest['init']
): Promise<any> {
  const response: Response = await fetch(url, addAuthToRequest(init))
  if (response.status === 401) {
    if (refreshing) {
      return new Promise(resolve => {
        queue.push({ url, init, resolve })
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
        q?.resolve(requestWithAuth(q.url, q.init))
      }

      return requestWithAuth(url, init)
    } else {
      refreshing = false
      queue.length = 0
      clearAuthToken()
    }
  } else {
    return init?.format === 'text' ? response.text() : response.json()
  }
}

function addAuthToRequest(
  init?: PendingRequest['init']
): PendingRequest['init'] {
  const headers = new Headers(init?.headers)

  if (typeof window !== 'undefined') {
    headers.append('authorization', `Bearer ${getAuthToken()?.accessToken}`)
  }

  return {
    ...(init || {}),
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

function requestWithAuth(url: string, init?: PendingRequest['init']) {
  return fetch(url, addAuthToRequest(init)).then(x => {
    return init?.format === 'text' ? x.text() : x.json()
  })
}
