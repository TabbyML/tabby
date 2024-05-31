import * as React from 'react'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import useLocalStorage from 'use-local-storage'

import { graphql } from '@/lib/gql/generates'

import { useIsAdminInitialized } from '../hooks/use-server-info'
import { useMutation } from './gql'
import { AUTH_TOKEN_KEY } from './token-management'

interface AuthData {
  accessToken: string
  refreshToken: string
}

function isSameAuthData(lhs: AuthData | undefined, rhs: AuthData | undefined) {
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
      data: undefined
    }

function isSameAuthState(lhs: AuthState, rhs: AuthState) {
  return lhs.status == rhs.status && isSameAuthData(lhs.data, rhs.data)
}

export enum AuthActionType {
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
        data: undefined
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

const refreshTokenMutation = graphql(/* GraphQL */ `
  mutation refreshToken($refreshToken: String!) {
    refreshToken(refreshToken: $refreshToken) {
      accessToken
      refreshToken
    }
  }
`)

const logoutAllSessionsMutation = graphql(/* GraphQL */ `
  mutation LogoutAllSessions {
    logoutAllSessions
  }
`)

const AuthProvider: React.FunctionComponent<AuthProviderProps> = ({
  children
}) => {
  const [initialized, setInitialized] = React.useState(false)
  const [authToken] = useLocalStorage<AuthData | undefined>(
    AUTH_TOKEN_KEY,
    undefined
  )
  const [authState, dispatch] = React.useReducer(authReducerDeduped, {
    status: 'loading',
    data: undefined
  })

  React.useEffect(() => {
    if (authToken?.accessToken && authToken?.refreshToken) {
      dispatch({ type: AuthActionType.SignIn, data: authToken })
    } else {
      dispatch({ type: AuthActionType.SignOut })
    }
    setInitialized(true)
  }, [])

  React.useEffect(() => {
    if (!initialized) return
    // After being mounted, listen for changes in the access token
    if (authToken?.accessToken && authToken?.refreshToken) {
      dispatch({ type: AuthActionType.Refresh, data: authToken })
    } else if (!authToken?.accessToken && !authToken?.refreshToken) {
      dispatch({ type: AuthActionType.SignOut })
    }
  }, [authToken])

  const session: Session = React.useMemo(() => {
    if (authState?.status == 'authenticated') {
      return {
        data: {
          accessToken: authState.data.accessToken
        },
        status: authState.status
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

export function useAuthStore() {
  const context = React.useContext(AuthContext)

  if (!context) {
    throw new AuthProviderIsMissing()
  }

  return context
}

function useSignIn(): (params: AuthData) => Promise<boolean> {
  const { dispatch } = useAuthStore()
  const [authToken, setAuthToken] = useLocalStorage<AuthData | undefined>(
    AUTH_TOKEN_KEY,
    undefined
  )
  return async data => {
    setAuthToken({
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
  const logoutAllSessions = useMutation(logoutAllSessionsMutation)
  const { dispatch } = useAuthStore()
  const [authToken, setAuthToken] = useLocalStorage<AuthData | undefined>(
    AUTH_TOKEN_KEY,
    undefined
  )
  return async () => {
    await logoutAllSessions()
    setAuthToken(undefined)
    dispatch({ type: AuthActionType.SignOut })
  }
}

interface JWTInfo {
  accessToken: string
}

type Session =
  | {
      data: null
      status: 'loading' | 'unauthenticated'
    }
  | {
      data: JWTInfo
      status: 'authenticated'
    }

function useSession(): Session {
  const { session } = useAuthStore()
  return session
}

const redirectWhitelist = [
  '/auth/signin',
  '/auth/signup',
  '/auth/reset-password'
]

function useAuthenticatedSession() {
  const isAdminInitialized = useIsAdminInitialized()
  const router = useRouter()
  const pathName = usePathname()
  const searchParams = useSearchParams()
  const { data: session, status } = useSession()

  React.useEffect(() => {
    if (status === 'loading') return
    if (status === 'authenticated') return
    if (isAdminInitialized === undefined) return

    const isAdminSignup =
      pathName === '/auth/signup' && searchParams.get('isAdmin') === 'true'

    if (!isAdminSignup && !isAdminInitialized) {
      return router.replace('/auth/signup?isAdmin=true')
    }

    if (!redirectWhitelist.includes(pathName)) {
      router.replace('/auth/signin')
    }
  }, [isAdminInitialized, status])

  return session
}

function useAuthenticatedApi(path: string | null): string | null {
  const { status } = useSession()
  return path && status === 'authenticated' ? path : null
}

export type { AuthStore, JWTInfo, Session, AuthData }

export {
  AuthProvider,
  useSignIn,
  useSignOut,
  useSession,
  useAuthenticatedSession,
  useAuthenticatedApi,
  AUTH_TOKEN_KEY,
  refreshTokenMutation,
  logoutAllSessionsMutation
}
