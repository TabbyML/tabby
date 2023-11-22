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
}

export type Mutation = {
  __typename?: 'Mutation'
  resetRegistrationToken: Scalars['String']['output']
}

export type Query = {
  __typename?: 'Query'
  registrationToken: Scalars['String']['output']
  workers: Array<Worker>
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

export type GetRegistrationTokenQueryVariables = Exact<{ [key: string]: never }>

export type GetRegistrationTokenQuery = {
  __typename?: 'Query'
  registrationToken: string
}

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
