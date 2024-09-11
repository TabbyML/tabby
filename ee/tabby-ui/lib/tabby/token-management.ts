import { jwtDecode } from 'jwt-decode'
import { isNil } from 'lodash-es'
import { FetcherOptions } from 'tabby-chat-panel/index'

import { isClientSide } from '../utils'
import { AuthData } from './auth'

export const AUTH_TOKEN_KEY = '_tabby_auth'
export const AUTH_LOCK_KEY = '_tabby_auth_lock'
const FETCHER_OPTIONS_KEY = '_tabby_chat_sdk_fetcher_options'

const getAuthToken = (): AuthData | undefined => {
  if (isClientSide()) {
    let tokenData = localStorage.getItem(AUTH_TOKEN_KEY)
    if (!tokenData) return undefined
    try {
      return JSON.parse(tokenData)
    } catch (e) {
      return undefined
    }
  }
  return undefined
}

const saveAuthToken = (authData: AuthData) => {
  localStorage.setItem(AUTH_TOKEN_KEY, JSON.stringify(authData))
}

const clearAuthToken = () => {
  localStorage.removeItem(AUTH_TOKEN_KEY)
  // FIXME(liangfung)
  // dispatching storageEvent to notify updating `authToken` in `AuthProvider`,
  // the `useEffect` hook depending on `authToken` in `AuthProvider` will be fired and updating the authState
  window.dispatchEvent(
    new StorageEvent('storage', {
      storageArea: window.localStorage,
      url: window.location.href,
      key: AUTH_TOKEN_KEY
    })
  )
}

const isTokenExpired = (exp: number | undefined): boolean => {
  return isNil(exp) ? true : Date.now() > exp * 1000
}

// Checks if the JWT token's issued-at time (iat) is within the last minute.
const isTokenRecentlyIssued = (iat: number | undefined): boolean => {
  return isNil(iat) ? false : Date.now() - iat * 1000 < 60 * 1000
}

export const getFetcherOptions = (): FetcherOptions | undefined => {
  try {
    let fetcherOptions = sessionStorage.getItem(FETCHER_OPTIONS_KEY)
    if (!fetcherOptions) return undefined
    return JSON.parse(fetcherOptions)
  } catch {
    return undefined
  }
}

export const clearFetcherOptions = () => {
  sessionStorage.removeItem(FETCHER_OPTIONS_KEY)
}

export const saveFetcherOptions = (fetcherOptions: FetcherOptions) => {
  if (!fetcherOptions) return
  try {
    sessionStorage.setItem(FETCHER_OPTIONS_KEY, JSON.stringify(fetcherOptions))
  } catch (e) {
    sessionStorage.removeItem(FETCHER_OPTIONS_KEY)
  }
}

class TokenManager {
  clearAccessToken() {
    let authToken = getAuthToken()
    if (authToken) {
      // remove accessToken only, keep the refreshToken
      saveAuthToken({
        ...authToken,
        accessToken: ''
      })
    }
  }

  async refreshToken(doRefreshToken: () => Promise<AuthData | undefined>) {
    try {
      if (typeof navigator?.locks === 'undefined') {
        // eslint-disable-next-line no-console
        console.error(
          'The Web Locks API is not supported in your browser. Please upgrade to a newer browser version.'
        )
        throw new Error()
      }

      await navigator.locks.request(AUTH_LOCK_KEY, async () => {
        const authToken = getAuthToken()
        const accessToken = authToken?.accessToken
        const refreshToken = authToken?.refreshToken

        let newAuthToken: AuthData | undefined

        if (accessToken) {
          const { iat } = jwtDecode(accessToken)
          if (isTokenRecentlyIssued(iat)) {
            newAuthToken = authToken
          } else {
            newAuthToken = await doRefreshToken()
          }
        } else if (refreshToken) {
          newAuthToken = await doRefreshToken()
        }

        if (newAuthToken) {
          saveAuthToken(newAuthToken)
        } else {
          clearAuthToken()
        }
      })
    } catch (e) {
      clearAuthToken()
    }
  }
}

const tokenManager = new TokenManager()

export {
  tokenManager,
  getAuthToken,
  saveAuthToken,
  clearAuthToken,
  isTokenExpired
}
