import { jwtDecode } from 'jwt-decode'
import { isNil } from 'lodash-es'

import { isClientSide } from '../utils'
import { AuthData } from './auth'

export const AUTH_TOKEN_KEY = '_tabby_auth'
export const AUTH_LOCK_KEY = '_tabby_auth_lock'

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

class TokenManager {
  async refreshToken(doRefreshToken: () => Promise<AuthData | undefined>) {
    try {
      if (typeof navigator?.locks === 'undefined') {
        console.error(
          'The Web Locks API is not supported in your browser. Please upgrade to a newer browser version.'
        )
        throw new Error()
      }

      await navigator.locks.request(AUTH_LOCK_KEY, async () => {
        const authToken = getAuthToken()
        const accessToken = getAuthToken()?.accessToken

        let newAuthToken: AuthData | undefined

        if (accessToken) {
          const { iat } = jwtDecode(accessToken)
          if (isTokenRecentlyIssued(iat)) {
            newAuthToken = authToken
          } else {
            newAuthToken = await doRefreshToken()
          }
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
