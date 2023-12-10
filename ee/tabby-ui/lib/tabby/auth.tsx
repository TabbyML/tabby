import * as React from 'react'
import { useRouter } from 'next/navigation'
import { jwtDecode, JwtPayload } from 'jwt-decode'

import { graphql } from '@/lib/gql/generates'
import useInterval from '@/lib/hooks/use-interval'
import { useGraphQLQuery, useMutation } from '@/lib/tabby/gql'

interface AuthData {
  accessToken: string
  refreshToken: string
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

enum AuthActionType {
  Init,
  SignIn,
  SignOut,
  Refresh
}

interface InitAction {
  type: AuthActionType.Init
  data: AuthData | null
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

type AuthActions = InitAction | SignInAction | SignOutAction | RefreshAction

function authReducer(state: AuthState, action: AuthActions): AuthState {
  switch (action.type) {
    case AuthActionType.Init:
    case AuthActionType.SignIn:
    case AuthActionType.Refresh:
      if (action.data) {
        return {
          status: 'authenticated',
          data: action.data
        }
      } else {
        return {
          status: 'unauthenticated',
          data: null
        }
      }
    case AuthActionType.SignOut:
      TokenStorage.reset()
      return {
        status: 'unauthenticated',
        data: null
      }
  }
}

class TokenStorage {
  static authName = '_tabby_auth'

  initialState(): AuthData | null {
    const authData = localStorage.getItem(TokenStorage.authName)
    if (authData) {
      return JSON.parse(authData)
    } else {
      return null
    }
  }

  persist(state: AuthData) {
    localStorage.setItem(TokenStorage.authName, JSON.stringify(state))
  }

  static reset() {
    localStorage.removeItem(TokenStorage.authName)
  }
}

interface AuthProviderProps {
  children: React.ReactNode
}

interface AuthStore {
  authState: AuthState | null
  dispatch: React.Dispatch<AuthActions>
}

const AuthContext = React.createContext<AuthStore | null>(null)

const refreshTokenMutation = graphql(/* GraphQL */ `
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
  const [authState, dispatch] = React.useReducer(authReducer, {
    status: 'loading',
    data: null
  })

  return (
    <AuthContext.Provider value={{ authState, dispatch }}>
      <RefreshAuth />
      {children}
    </AuthContext.Provider>
  )
}

function RefreshAuth() {
  const { authState, dispatch } = useAuthStore()
  const storage = new TokenStorage()
  const refreshToken = useMutation(refreshTokenMutation, {
    onCompleted({ refreshToken: data }) {
      dispatch({ type: AuthActionType.Refresh, data })
    },
    onError() {
      dispatch({
        type: AuthActionType.SignOut
      })
    }
  })

  const initialized = React.useRef(false)
  React.useEffect(() => {
    if (initialized.current) return

    initialized.current = true
    const data = storage.initialState()
    if (data?.refreshToken) {
      refreshToken(data)
    } else {
      dispatch({ type: AuthActionType.Init, data: null })
    }
  }, [])

  React.useEffect(() => {
    authState?.data && storage.persist(authState.data)
  }, [authState])

  useInterval(async () => {
    if (authState?.status !== 'authenticated') {
      return
    }

    await refreshToken(authState.data)
  }, 5)

  return <></>
}

class AuthProviderIsMissing extends Error {
  constructor() {
    super('AuthProvider is missing. Please add the AuthProvider at root level')
  }
}

function useAuthStore(): AuthStore {
  const context = React.useContext(AuthContext)

  if (!context) {
    throw new AuthProviderIsMissing()
  }

  return context
}

function useSignIn(): (params: AuthData) => Promise<boolean> {
  const { dispatch } = useAuthStore()
  return async data => {
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
  const { authState } = useAuthStore()
  if (authState?.status == 'authenticated') {
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
  } else {
    return {
      status: authState?.status ?? 'loading',
      data: null
    }
  }
}

export const getIsAdminInitialized = graphql(/* GraphQL */ `
  query GetIsAdminInitialized {
    isAdminInitialized
  }
`)

function useAuthenticatedSession() {
  const { data } = useGraphQLQuery(getIsAdminInitialized)
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

function useAuthenticatedApi(path: string | null): [string, string] | null {
  const { data, status } = useSession()
  return path && status === 'authenticated' ? [path, data.accessToken] : null
}

export type { AuthStore, User, Session }

export {
  AuthProvider,
  useSignIn,
  useSignOut,
  useSession,
  useAuthenticatedSession,
  useAuthenticatedApi
}
