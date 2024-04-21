import { isClientSide } from '../utils'
import { AuthData } from './auth'

export const AUTH_TOKEN_KEY = '_tabby_auth'
export const AUTH_LOCK_KEY = '_tabby_auth_lock'
export const AUTH_LOCK_TIMEOUT = 1000 * 10

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
  private broadcastChannel: BroadcastChannel | null
  refreshPromise: Promise<AuthData | undefined> | null

  constructor() {
    this.refreshPromise = null
    this.retryQueue = []
    this.broadcastChannel =
      typeof window !== 'undefined' ? new BroadcastChannel(AUTH_LOCK_KEY) : null

    this.broadcastChannel?.addEventListener('message', e =>
      this.onRefreshCompleteReceived(e)
    )
  }

  private notifyRefreshComplete(error?: any) {
    this.broadcastChannel?.postMessage({
      success: !error,
      error
    })
  }

  onRefreshCompleteReceived = (event: MessageEvent<any>) => {
    if (event.data?.success) {
      this.processQueue()
    } else {
      this.rejectQueue(event.data?.error)
    }
  }

  enqueueRetryRequest() {
    return new Promise<void>((resolve, reject) => {
      this.retryQueue.push((success: boolean, error?: Error) => {
        if (!success || error) {
          reject(error ?? 'Failed to refresh token')
        } else {
          resolve()
        }
      })
    })
  }

  processQueue() {
    this.refreshPromise = null
    if (this.retryQueue.length) {
      this.retryQueue.forEach(retryCallback => retryCallback(true))
    }
    this.retryQueue = []
  }

  rejectQueue(error?: Error) {
    this.refreshPromise = null
    if (this.retryQueue?.length) {
      this.retryQueue.forEach(retryCallback => retryCallback(false, error))
    }
    this.retryQueue = []
  }

  async tryGetRefreshLock() {
    try {
      const locks = await navigator.locks.query()
      const refreshLock = locks.held?.some(lock => lock.name === AUTH_LOCK_KEY)
      return !!refreshLock
    } catch (e) {
      return false
    }
  }

  async refreshToken(doRefreshToken: () => Promise<AuthData | undefined>) {
    // if refreshing, enquqeRetryRequest
    if (this.refreshPromise) {
      return this.enqueueRetryRequest()
    }

    // if it's refreshing token in other processes, enquqeRetryRequest
    if (await this.tryGetRefreshLock()) {
      return this.enqueueRetryRequest()
    }

    const abortController = new AbortController()
    let timeoutId = window.setTimeout(() => {
      abortController.abort()
    }, AUTH_LOCK_TIMEOUT)

    try {
      this.refreshPromise = navigator.locks.request(
        AUTH_LOCK_KEY,
        { signal: abortController.signal },
        async () => {
          const token = await doRefreshToken()
          if (token) {
            saveAuthToken(token)
            this.processQueue()
          } else {
            clearAuthToken()
            this.rejectQueue()
          }
        }
      )
      await this.refreshPromise

      this.notifyRefreshComplete()
    } catch (e) {
      clearAuthToken()
      this.rejectQueue()
      this.notifyRefreshComplete(e)
    } finally {
      clearTimeout(timeoutId)
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
