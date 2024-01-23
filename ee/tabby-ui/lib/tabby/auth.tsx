import * as React from 'react'
import { useRouter } from 'next/navigation'
import { jwtDecode, JwtPayload } from 'jwt-decode'
import { useQuery } from 'urql'
import useLocalStorage from 'use-local-storage'

import { graphql } from '@/lib/gql/generates'
import { isClientSide } from '@/lib/utils'

interface AuthData {
  accessToken: string
  refreshToken: string
}

function isSameAuthData(lhs: AuthData | null, rhs: AuthData | null) {
  return (
    lhs?.accessToken === rhs?.accessToken &&
    lhs?.refreshToken === rhs?.refreshToken
  )
}

type AuthState =
  | {
      status: 'authenticated'
      data: AuthData
    }
  | {
      status: 'loading' | 'unauthenticated'
      data: null
    }

function isSameAuthState(lhs: AuthState, rhs: AuthState) {
  return lhs.status == rhs.status && isSameAuthData(lhs.data, rhs.data)
}

enum AuthActionType {
  SignIn,
  SignOut,
  Refresh
}

interface SignInAction {
  type: AuthActionType.SignIn
  data: AuthData
}

interface SignOutAction {
  type: AuthActionType.SignOut
}

interface RefreshAction {
  type: AuthActionType.Refresh
  data: AuthData
}

type AuthActions = SignInAction | SignOutAction | RefreshAction

const AUTH_TOKEN_KEY = '_tabby_auth'

const getAuthToken = (): AuthData | null => {
  if (isClientSide()) {
    let tokenData = localStorage.getItem(AUTH_TOKEN_KEY)
    if (!tokenData) return null
    try {
      return JSON.parse(tokenData)
    } catch (e) {
      return null
    }
  }
  return null
}
const saveAuthToken = (authData: AuthData) => {
  localStorage.setItem(AUTH_TOKEN_KEY, JSON.stringify(authData))
}
const clearAuthToken = () => {
  localStorage.removeItem(AUTH_TOKEN_KEY)
}

function authReducer(state: AuthState, action: AuthActions): AuthState {
  switch (action.type) {
    case AuthActionType.SignIn:
    case AuthActionType.Refresh:
      return {
        status: 'authenticated',
        data: action.data
      }
    case AuthActionType.SignOut:
      return {
        status: 'unauthenticated',
        data: null
      }
  }
}

function authReducerDeduped(state: AuthState, action: AuthActions): AuthState {
  const newState = authReducer(state, action)
  if (isSameAuthState(state, newState)) {
    return state
  } else {
    return newState
  }
}

interface AuthProviderProps {
  children: React.ReactNode
}

interface AuthContextValue extends AuthStore {
  session: Session
}

interface AuthStore {
  authState: AuthState | null
  dispatch: React.Dispatch<AuthActions>
}

const AuthContext = React.createContext<AuthContextValue>(
  {} as AuthContextValue
)

export const refreshTokenMutation = graphql(/* GraphQL */ `
  mutation refreshToken($refreshToken: String!) {
    refreshToken(refreshToken: $refreshToken) {
      accessToken
      refreshToken
    }
  }
`)

const AuthProvider: React.FunctionComponent<AuthProviderProps> = ({
  children
}) => {
  const [authToken] = useLocalStorage<AuthData | null>(AUTH_TOKEN_KEY, null)
  const [authState, dispatch] = React.useReducer(authReducerDeduped, {
    status: 'loading',
    data: null
  })
  React.useEffect(() => {
    if (authToken?.accessToken && authToken?.refreshToken) {
      dispatch({ type: AuthActionType.Refresh, data: authToken })
    } else {
      dispatch({ type: AuthActionType.SignOut })
    }
  }, [authToken])

  const session: Session = React.useMemo(() => {
    if (authState?.status == 'authenticated') {
      try {
        const { sub, is_admin } = jwtDecode<JwtPayload & { is_admin: boolean }>(
          authState.data.accessToken
        )
        return {
          data: {
            email: sub!,
            isAdmin: is_admin,
            accessToken: authState.data.accessToken
          },
          status: authState.status
        }
      } catch (e) {
        console.error('jwt decode failed')
        return {
          status: authState?.status ?? 'loading',
          data: {
            email: '',
            isAdmin: false,
            accessToken: authState.data.accessToken
          }
        }
      }
    }

    return {
      status: authState?.status ?? 'loading',
      data: null
    }
  }, [authState])

  return (
    <AuthContext.Provider value={{ authState, dispatch, session }}>
      {children}
    </AuthContext.Provider>
  )
}

class AuthProviderIsMissing extends Error {
  constructor() {
    super('AuthProvider is missing. Please add the AuthProvider at root level')
  }
}

function useAuthStore() {
  const context = React.useContext(AuthContext)

  if (!context) {
    throw new AuthProviderIsMissing()
  }

  return context
}

function useSignIn(): (params: AuthData) => Promise<boolean> {
  const { dispatch } = useAuthStore()
  return async data => {
    saveAuthToken({
      accessToken: data.accessToken,
      refreshToken: data.refreshToken
    })
    dispatch({
      type: AuthActionType.SignIn,
      data
    })

    return true
  }
}

function useSignOut(): () => Promise<void> {
  const { dispatch } = useAuthStore()
  return async () => {
    clearAuthToken()
    dispatch({ type: AuthActionType.SignOut })
  }
}

interface User {
  email: string
  isAdmin: boolean
  accessToken: string
}

type Session =
  | {
      data: null
      status: 'loading' | 'unauthenticated'
    }
  | {
      data: User
      status: 'authenticated'
    }

function useSession(): Session {
  const { session } = useAuthStore()
  return session
}

export const getIsAdminInitialized = graphql(/* GraphQL */ `
  query GetIsAdminInitialized {
    isAdminInitialized
  }
`)

function useAuthenticatedSession() {
  const [{ data }] = useQuery({ query: getIsAdminInitialized })
  const router = useRouter()
  const { data: session, status } = useSession()

  React.useEffect(() => {
    if (status === 'loading') return
    if (status === 'authenticated') return

    if (data?.isAdminInitialized === false) {
      router.replace('/auth/signup?isAdmin=true')
    } else if (status === 'unauthenticated') {
      router.replace('/auth/signin')
    }
  }, [data, status])

  return session
}

function useAuthenticatedApi(path: string | null): string | null {
  const { status } = useSession()
  return path && status === 'authenticated' ? path : null
}

export type { AuthStore, User, Session }

export {
  AuthProvider,
  useSignIn,
  useSignOut,
  useSession,
  useAuthenticatedSession,
  useAuthenticatedApi,
  getAuthToken,
  saveAuthToken,
  clearAuthToken,
  AUTH_TOKEN_KEY
}
