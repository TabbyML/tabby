/* eslint-disable */
import { TypedDocumentNode as DocumentNode } from '@graphql-typed-document-node/core'

export type Maybe<T> = T | null
export type InputMaybe<T> = Maybe<T>
export type Exact<T extends { [key: string]: unknown }> = {
  [K in keyof T]: T[K]
}
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & {
  [SubKey in K]?: Maybe<T[SubKey]>
}
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & {
  [SubKey in K]: Maybe<T[SubKey]>
}
export type MakeEmpty<
  T extends { [key: string]: unknown },
  K extends keyof T
> = { [_ in K]?: never }
export type Incremental<T> =
  | T
  | {
      [P in keyof T]?: P extends ' $fragmentName' | '__typename' ? T[P] : never
    }
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: { input: string; output: string }
  String: { input: string; output: string }
  Boolean: { input: boolean; output: boolean }
  Int: { input: number; output: number }
  Float: { input: number; output: number }
  /** DateTime */
  DateTimeUtc: { input: any; output: any }
}

export type Invitation = {
  __typename?: 'Invitation'
  code: Scalars['String']['output']
  createdAt: Scalars['String']['output']
  email: Scalars['String']['output']
  id: Scalars['Int']['output']
}

export type JwtPayload = {
  __typename?: 'JWTPayload'
  /** Expiration time (as UTC timestamp) */
  exp: Scalars['Float']['output']
  /** Issued at (as UTC timestamp) */
  iat: Scalars['Float']['output']
  /** Whether the user is admin. */
  isAdmin: Scalars['Boolean']['output']
  /** User email address */
  sub: Scalars['String']['output']
}

export type Mutation = {
  __typename?: 'Mutation'
  createInvitation: Scalars['Int']['output']
  deleteInvitation: Scalars['Int']['output']
  refreshToken: RefreshTokenResponse
  register: RegisterResponse
  resetRegistrationToken: Scalars['String']['output']
  tokenAuth: TokenAuthResponse
  verifyToken: VerifyTokenResponse
}

export type MutationCreateInvitationArgs = {
  email: Scalars['String']['input']
}

export type MutationDeleteInvitationArgs = {
  id: Scalars['Int']['input']
}

export type MutationRefreshTokenArgs = {
  refreshToken: Scalars['String']['input']
}

export type MutationRegisterArgs = {
  email: Scalars['String']['input']
  invitationCode?: InputMaybe<Scalars['String']['input']>
  password1: Scalars['String']['input']
  password2: Scalars['String']['input']
}

export type MutationTokenAuthArgs = {
  email: Scalars['String']['input']
  password: Scalars['String']['input']
}

export type MutationVerifyTokenArgs = {
  token: Scalars['String']['input']
}

export type Query = {
  __typename?: 'Query'
  invitations: Array<Invitation>
  isAdminInitialized: Scalars['Boolean']['output']
  me: User
  registrationToken: Scalars['String']['output']
  workers: Array<Worker>
}

export type RefreshTokenResponse = {
  __typename?: 'RefreshTokenResponse'
  accessToken: Scalars['String']['output']
  refreshExpiresAt: Scalars['DateTimeUtc']['output']
  refreshToken: Scalars['String']['output']
}

export type RegisterResponse = {
  __typename?: 'RegisterResponse'
  accessToken: Scalars['String']['output']
  refreshToken: Scalars['String']['output']
}

export type TokenAuthResponse = {
  __typename?: 'TokenAuthResponse'
  accessToken: Scalars['String']['output']
  refreshToken: Scalars['String']['output']
}

export type User = {
  __typename?: 'User'
  authToken: Scalars['String']['output']
  email: Scalars['String']['output']
  isAdmin: Scalars['Boolean']['output']
}

export type VerifyTokenResponse = {
  __typename?: 'VerifyTokenResponse'
  claims: JwtPayload
}

export type Worker = {
  __typename?: 'Worker'
  addr: Scalars['String']['output']
  arch: Scalars['String']['output']
  cpuCount: Scalars['Int']['output']
  cpuInfo: Scalars['String']['output']
  cudaDevices: Array<Scalars['String']['output']>
  device: Scalars['String']['output']
  kind: WorkerKind
  name: Scalars['String']['output']
}

export enum WorkerKind {
  Chat = 'CHAT',
  Completion = 'COMPLETION'
}

export type GetRegistrationTokenQueryVariables = Exact<{ [key: string]: never }>

export type GetRegistrationTokenQuery = {
  __typename?: 'Query'
  registrationToken: string
}

export type TokenAuthMutationVariables = Exact<{
  email: Scalars['String']['input']
  password: Scalars['String']['input']
}>

export type TokenAuthMutation = {
  __typename?: 'Mutation'
  tokenAuth: {
    __typename?: 'TokenAuthResponse'
    accessToken: string
    refreshToken: string
  }
}

export type RegisterMutationVariables = Exact<{
  email: Scalars['String']['input']
  password1: Scalars['String']['input']
  password2: Scalars['String']['input']
  invitationCode?: InputMaybe<Scalars['String']['input']>
}>

export type RegisterMutation = {
  __typename?: 'Mutation'
  register: {
    __typename?: 'RegisterResponse'
    accessToken: string
    refreshToken: string
  }
}

export type GetWorkersQueryVariables = Exact<{ [key: string]: never }>

export type GetWorkersQuery = {
  __typename?: 'Query'
  workers: Array<{
    __typename?: 'Worker'
    kind: WorkerKind
    name: string
    addr: string
    device: string
    arch: string
    cpuInfo: string
    cpuCount: number
    cudaDevices: Array<string>
  }>
}

export type RefreshTokenMutationVariables = Exact<{
  refreshToken: Scalars['String']['input']
}>

export type RefreshTokenMutation = {
  __typename?: 'Mutation'
  refreshToken: {
    __typename?: 'RefreshTokenResponse'
    accessToken: string
    refreshToken: string
  }
}

export type GetIsAdminInitializedQueryVariables = Exact<{
  [key: string]: never
}>

export type GetIsAdminInitializedQuery = {
  __typename?: 'Query'
  isAdminInitialized: boolean
}

export const GetRegistrationTokenDocument = {
  kind: 'Document',
  definitions: [
    {
      kind: 'OperationDefinition',
      operation: 'query',
      name: { kind: 'Name', value: 'GetRegistrationToken' },
      selectionSet: {
        kind: 'SelectionSet',
        selections: [
          { kind: 'Field', name: { kind: 'Name', value: 'registrationToken' } }
        ]
      }
    }
  ]
} as unknown as DocumentNode<
  GetRegistrationTokenQuery,
  GetRegistrationTokenQueryVariables
>
export const TokenAuthDocument = {
  kind: 'Document',
  definitions: [
    {
      kind: 'OperationDefinition',
      operation: 'mutation',
      name: { kind: 'Name', value: 'tokenAuth' },
      variableDefinitions: [
        {
          kind: 'VariableDefinition',
          variable: {
            kind: 'Variable',
            name: { kind: 'Name', value: 'email' }
          },
          type: {
            kind: 'NonNullType',
            type: { kind: 'NamedType', name: { kind: 'Name', value: 'String' } }
          }
        },
        {
          kind: 'VariableDefinition',
          variable: {
            kind: 'Variable',
            name: { kind: 'Name', value: 'password' }
          },
          type: {
            kind: 'NonNullType',
            type: { kind: 'NamedType', name: { kind: 'Name', value: 'String' } }
          }
        }
      ],
      selectionSet: {
        kind: 'SelectionSet',
        selections: [
          {
            kind: 'Field',
            name: { kind: 'Name', value: 'tokenAuth' },
            arguments: [
              {
                kind: 'Argument',
                name: { kind: 'Name', value: 'email' },
                value: {
                  kind: 'Variable',
                  name: { kind: 'Name', value: 'email' }
                }
              },
              {
                kind: 'Argument',
                name: { kind: 'Name', value: 'password' },
                value: {
                  kind: 'Variable',
                  name: { kind: 'Name', value: 'password' }
                }
              }
            ],
            selectionSet: {
              kind: 'SelectionSet',
              selections: [
                { kind: 'Field', name: { kind: 'Name', value: 'accessToken' } },
                { kind: 'Field', name: { kind: 'Name', value: 'refreshToken' } }
              ]
            }
          }
        ]
      }
    }
  ]
} as unknown as DocumentNode<TokenAuthMutation, TokenAuthMutationVariables>
export const RegisterDocument = {
  kind: 'Document',
  definitions: [
    {
      kind: 'OperationDefinition',
      operation: 'mutation',
      name: { kind: 'Name', value: 'register' },
      variableDefinitions: [
        {
          kind: 'VariableDefinition',
          variable: {
            kind: 'Variable',
            name: { kind: 'Name', value: 'email' }
          },
          type: {
            kind: 'NonNullType',
            type: { kind: 'NamedType', name: { kind: 'Name', value: 'String' } }
          }
        },
        {
          kind: 'VariableDefinition',
          variable: {
            kind: 'Variable',
            name: { kind: 'Name', value: 'password1' }
          },
          type: {
            kind: 'NonNullType',
            type: { kind: 'NamedType', name: { kind: 'Name', value: 'String' } }
          }
        },
        {
          kind: 'VariableDefinition',
          variable: {
            kind: 'Variable',
            name: { kind: 'Name', value: 'password2' }
          },
          type: {
            kind: 'NonNullType',
            type: { kind: 'NamedType', name: { kind: 'Name', value: 'String' } }
          }
        },
        {
          kind: 'VariableDefinition',
          variable: {
            kind: 'Variable',
            name: { kind: 'Name', value: 'invitationCode' }
          },
          type: { kind: 'NamedType', name: { kind: 'Name', value: 'String' } }
        }
      ],
      selectionSet: {
        kind: 'SelectionSet',
        selections: [
          {
            kind: 'Field',
            name: { kind: 'Name', value: 'register' },
            arguments: [
              {
                kind: 'Argument',
                name: { kind: 'Name', value: 'email' },
                value: {
                  kind: 'Variable',
                  name: { kind: 'Name', value: 'email' }
                }
              },
              {
                kind: 'Argument',
                name: { kind: 'Name', value: 'password1' },
                value: {
                  kind: 'Variable',
                  name: { kind: 'Name', value: 'password1' }
                }
              },
              {
                kind: 'Argument',
                name: { kind: 'Name', value: 'password2' },
                value: {
                  kind: 'Variable',
                  name: { kind: 'Name', value: 'password2' }
                }
              },
              {
                kind: 'Argument',
                name: { kind: 'Name', value: 'invitationCode' },
                value: {
                  kind: 'Variable',
                  name: { kind: 'Name', value: 'invitationCode' }
                }
              }
            ],
            selectionSet: {
              kind: 'SelectionSet',
              selections: [
                { kind: 'Field', name: { kind: 'Name', value: 'accessToken' } },
                { kind: 'Field', name: { kind: 'Name', value: 'refreshToken' } }
              ]
            }
          }
        ]
      }
    }
  ]
} as unknown as DocumentNode<RegisterMutation, RegisterMutationVariables>
export const GetWorkersDocument = {
  kind: 'Document',
  definitions: [
    {
      kind: 'OperationDefinition',
      operation: 'query',
      name: { kind: 'Name', value: 'GetWorkers' },
      selectionSet: {
        kind: 'SelectionSet',
        selections: [
          {
            kind: 'Field',
            name: { kind: 'Name', value: 'workers' },
            selectionSet: {
              kind: 'SelectionSet',
              selections: [
                { kind: 'Field', name: { kind: 'Name', value: 'kind' } },
                { kind: 'Field', name: { kind: 'Name', value: 'name' } },
                { kind: 'Field', name: { kind: 'Name', value: 'addr' } },
                { kind: 'Field', name: { kind: 'Name', value: 'device' } },
                { kind: 'Field', name: { kind: 'Name', value: 'arch' } },
                { kind: 'Field', name: { kind: 'Name', value: 'cpuInfo' } },
                { kind: 'Field', name: { kind: 'Name', value: 'cpuCount' } },
                { kind: 'Field', name: { kind: 'Name', value: 'cudaDevices' } }
              ]
            }
          }
        ]
      }
    }
  ]
} as unknown as DocumentNode<GetWorkersQuery, GetWorkersQueryVariables>
export const RefreshTokenDocument = {
  kind: 'Document',
  definitions: [
    {
      kind: 'OperationDefinition',
      operation: 'mutation',
      name: { kind: 'Name', value: 'refreshToken' },
      variableDefinitions: [
        {
          kind: 'VariableDefinition',
          variable: {
            kind: 'Variable',
            name: { kind: 'Name', value: 'refreshToken' }
          },
          type: {
            kind: 'NonNullType',
            type: { kind: 'NamedType', name: { kind: 'Name', value: 'String' } }
          }
        }
      ],
      selectionSet: {
        kind: 'SelectionSet',
        selections: [
          {
            kind: 'Field',
            name: { kind: 'Name', value: 'refreshToken' },
            arguments: [
              {
                kind: 'Argument',
                name: { kind: 'Name', value: 'refreshToken' },
                value: {
                  kind: 'Variable',
                  name: { kind: 'Name', value: 'refreshToken' }
                }
              }
            ],
            selectionSet: {
              kind: 'SelectionSet',
              selections: [
                { kind: 'Field', name: { kind: 'Name', value: 'accessToken' } },
                { kind: 'Field', name: { kind: 'Name', value: 'refreshToken' } }
              ]
            }
          }
        ]
      }
    }
  ]
} as unknown as DocumentNode<
  RefreshTokenMutation,
  RefreshTokenMutationVariables
>
export const GetIsAdminInitializedDocument = {
  kind: 'Document',
  definitions: [
    {
      kind: 'OperationDefinition',
      operation: 'query',
      name: { kind: 'Name', value: 'GetIsAdminInitialized' },
      selectionSet: {
        kind: 'SelectionSet',
        selections: [
          { kind: 'Field', name: { kind: 'Name', value: 'isAdminInitialized' } }
        ]
      }
    }
  ]
} as unknown as DocumentNode<
  GetIsAdminInitializedQuery,
  GetIsAdminInitializedQueryVariables
>
