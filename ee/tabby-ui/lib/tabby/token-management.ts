import { isClientSide } from '../utils'
import { AuthData } from './auth'

export const AUTH_TOKEN_KEY = '_tabby_auth'
export const AUTH_LOCK_KEY = '_tabby_auth_lock'
export const AUTH_LOCK_EXP = 1000 * 10

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

const isTokenExpired = (exp: number) => {
  return Date.now() > exp * 1000
}

class TokenManager {
  private retryQueue: Array<(success: boolean, error?: Error) => void>

  constructor() {
    this.retryQueue = []

    if (typeof window !== 'undefined') {
      window.addEventListener('storage', this.handleStorageChange)
    }
  }

  private handleStorageChange = (event: StorageEvent) => {
    try {
      if (
        event.key === AUTH_LOCK_KEY &&
        event.newValue === null &&
        this.retryQueue?.length
      ) {
        this.processQueue()
      }
    } catch (e) {}
  }

  tryGetRefreshLock() {
    const currentLock = localStorage.getItem(AUTH_LOCK_KEY)
    const lockTimestamp = currentLock ? parseInt(currentLock, 10) : null
    const now = Date.now()
    if (
      !currentLock ||
      (lockTimestamp && now - lockTimestamp > AUTH_LOCK_EXP)
    ) {
      localStorage.setItem(AUTH_LOCK_KEY, now.toString())
      return true
    }
    return false
  }

  releaseRefreshLock() {
    localStorage.removeItem(AUTH_LOCK_KEY)
  }

  enqueueRetryRequest(
    retryCallback: (success: boolean, error?: Error) => void
  ) {
    this.retryQueue.push(retryCallback)
  }

  processQueue() {
    this.retryQueue.forEach(retryCallback => retryCallback(true))
    this.retryQueue = []
    this.releaseRefreshLock()
  }

  rejectQueue(error?: Error) {
    this.retryQueue.forEach(retryCallback => retryCallback(false, error))
    this.retryQueue = []
    this.releaseRefreshLock()
  }

  async refreshToken(doRefreshToken: () => Promise<AuthData | undefined>) {
    if (!this.tryGetRefreshLock()) {
      // refreshing
      return new Promise<void>((resolve, reject) => {
        this.enqueueRetryRequest((success: boolean, error?: Error) => {
          if (!success || error) {
            reject(error ?? 'Failed to refresh token')
          } else {
            resolve()
          }
        })
      })
    }

    const newToken = await doRefreshToken()
    if (newToken) {
      await saveAuthToken(newToken)
      this.processQueue()
    } else {
      this.rejectQueue()
      clearAuthToken()
      throw new Error('Failed to refresh token')
    }
  }
}

const tokenManagerInstance = new TokenManager()

export {
  tokenManagerInstance,
  getAuthToken,
  saveAuthToken,
  clearAuthToken,
  isTokenExpired
}
