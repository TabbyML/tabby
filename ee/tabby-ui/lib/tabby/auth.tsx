import * as React from 'react'
import { graphql } from '@/lib/gql/generates'
import useInterval from '@/lib/hooks/use-interval'
import { gqlClient } from '@/lib/tabby-gql-client'
import { jwtDecode } from 'jwt-decode'

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

async function doRefresh(token: string, dispatch: React.Dispatch<AuthActions>) {
  let action: AuthActions
  try {
    action = {
      type: AuthActionType.Refresh,
      data: (
        await gqlClient.request(refreshTokenMutation, { refreshToken: token })
      ).refreshToken
    }
  } catch (err) {
    console.error('Failed to refresh token', err)
    action = {
      type: AuthActionType.SignOut
    }
  }

  dispatch(action)
}

const AuthProvider: React.FunctionComponent<AuthProviderProps> = ({
  children
}) => {
  const storage = new TokenStorage()

  const [authState, dispatch] = React.useReducer(authReducer, {
    status: 'loading',
    data: null
  })

  React.useEffect(() => {
    const data = storage.initialState()
    if (data?.refreshToken) {
      doRefresh(data.refreshToken, dispatch)
    } else {
      dispatch({ type: AuthActionType.Init, data: null })
    }
  }, [])

  React.useEffect(() => {
    authState.data && storage.persist(authState.data)
  }, [authState])

  useInterval(async () => {
    if (authState.status !== 'authenticated') {
      return
    }

    await doRefresh(authState.data.refreshToken, dispatch)
  }, 5)

  return (
    <AuthContext.Provider value={{ authState, dispatch }}>
      {children}
    </AuthContext.Provider>
  )
}

function useAuthStore(): AuthStore {
  const context = React.useContext(AuthContext)

  if (!context) {
    throw new Error(
      'AuthProvider is missing. Please add the AuthProvider at root level'
    )
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
    const { user } = jwtDecode<{ user: User }>(authState.data.accessToken)
    return {
      data: user,
      status: authState.status
    }
  } else {
    return {
      status: authState?.status ?? 'loading',
      data: null
    }
  }
}

export type { AuthStore, User, Session }

export { AuthProvider, useSignIn, useSignOut, useSession }
