import { TypedDocumentNode } from '@graphql-typed-document-node/core'
import { authExchange } from '@urql/exchange-auth'
import { cacheExchange } from '@urql/exchange-graphcache'
import { relayPagination } from '@urql/exchange-graphcache/extras'
import { jwtDecode } from 'jwt-decode'
import { isNil } from 'lodash-es'
import { FieldValues, UseFormReturn } from 'react-hook-form'
import {
  AnyVariables,
  Client,
  CombinedError,
  errorExchange,
  fetchExchange,
  OperationResult,
  useMutation as useUrqlMutation
} from 'urql'

import {
  GitRepositoriesQueryVariables,
  ListInvitationsQueryVariables
} from '../gql/generates/graphql'
import { refreshTokenMutation } from './auth'
import { listInvitations, listRepositories } from './query'
import { getAuthToken, isTokenExpired, tokenManager } from './token-management'

interface ValidationError {
  path: string
  message: string
}

interface ValidationErrors {
  errors: Array<ValidationError>
}

function useMutation<TResult, TVariables extends AnyVariables>(
  document: TypedDocumentNode<TResult, TVariables>,
  options?: {
    onCompleted?: (data: TResult) => void
    onError?: (err: CombinedError) => any
    form?: any
  }
) {
  const [mutationResult, executeMutation] = useUrqlMutation<TResult>(document)
  const onFormError = options?.form
    ? makeFormErrorHandler(options.form)
    : undefined

  const fn = async (variables?: TVariables) => {
    let response: OperationResult<TResult, AnyVariables> | undefined

    try {
      response = await executeMutation(variables)
      if (response?.error) {
        onFormError && onFormError(response.error)
        options?.onError && options.onError(response.error)
      } else if (!isNil(response?.data)) {
        options?.onCompleted?.(response.data)
      }
    } catch (err: any) {
      options?.onError && options.onError(err)
      return
    }

    return response
  }

  return fn
}

function makeFormErrorHandler<T extends FieldValues>(form: UseFormReturn<T>) {
  return (err: CombinedError) => {
    const { graphQLErrors = [] } = err
    for (const error of graphQLErrors) {
      if (error.extensions && error.extensions['validation-errors']) {
        const validationErrors = error.extensions[
          'validation-errors'
        ] as ValidationErrors
        for (const error of validationErrors.errors) {
          form.setError(error.path as any, error)
        }
      } else if (error?.originalError) {
        form.setError('root', error.originalError)
      }
    }
  }
}

const client = new Client({
  url: `/graphql`,
  requestPolicy: 'cache-and-network',
  exchanges: [
    cacheExchange({
      keys: {
        CompletionStats: () => null,
        ServerInfo: () => null,
        RepositorySearch: () => null,
        RepositoryList: () => null
      },
      resolvers: {
        Query: {
          invitations: relayPagination(),
          repositories: relayPagination()
        }
      },
      updates: {
        Mutation: {
          deleteInvitation(result, args, cache, info) {
            if (result.deleteInvitation) {
              cache
                .inspectFields('Query')
                .filter(field => field.fieldName === 'invitations')
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: listInvitations,
                      variables:
                        field.arguments as ListInvitationsQueryVariables
                    },
                    data => {
                      if (data?.invitations?.edges) {
                        data.invitations.edges = data.invitations.edges.filter(
                          e => e.node.id !== args.id
                        )
                      }
                      return data
                    }
                  )
                })
            }
          },
          deleteGitRepository(result, args, cache, info) {
            if (result.deleteGitRepository) {
              cache
                .inspectFields('Query')
                .filter(field => field.fieldName === 'gitRepositories')
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: listRepositories,
                      variables:
                        field.arguments as GitRepositoriesQueryVariables
                    },
                    data => {
                      if (data?.gitRepositories?.edges) {
                        data.gitRepositories.edges =
                          data.gitRepositories.edges.filter(
                            e => e.node.id !== args.id
                          )
                      }
                      return data
                    }
                  )
                })
            }
          }
        }
      }
    }),
    authExchange(async utils => {
      const authData = getAuthToken()
      let accessToken = authData?.accessToken
      let refreshToken = authData?.refreshToken

      return {
        addAuthToOperation(operation) {
          // Sync tokens on every operation
          const authData = getAuthToken()
          accessToken = authData?.accessToken
          refreshToken = authData?.refreshToken
          if (!accessToken) return operation
          return utils.appendHeaders(operation, {
            Authorization: `Bearer ${accessToken}`
          })
        },
        didAuthError(error, _operation) {
          return error.graphQLErrors.some(
            e => e?.extensions?.code === 'UNAUTHORIZED'
          )
        },
        willAuthError(operation) {
          // Sync tokens on every operation
          const authData = getAuthToken()
          accessToken = authData?.accessToken
          refreshToken = authData?.refreshToken

          if (
            operation.kind === 'query' &&
            operation.query.definitions.some(definition => {
              return (
                definition.kind === 'OperationDefinition' &&
                definition.name?.value &&
                ['GetServerInfo'].includes(definition.name.value)
              )
            })
          ) {
            return false
          }

          if (
            operation.kind === 'mutation' &&
            operation.query.definitions.some(definition => {
              return (
                definition.kind === 'OperationDefinition' &&
                definition.name?.value &&
                ['tokenAuth', 'register'].includes(definition.name.value)
              )
            })
          ) {
            return false
          }

          if (
            refreshToken &&
            operation.kind === 'mutation' &&
            operation.query.definitions.some(definition => {
              return (
                definition.kind === 'OperationDefinition' &&
                definition?.name?.value === 'refreshToken'
              )
            })
          ) {
            return false
          }

          if (accessToken) {
            try {
              const { exp } = jwtDecode(accessToken)
              // Check whether `token` JWT is expired
              return isTokenExpired(exp)
            } catch (e) {
              return true
            }
          } else {
            return true
          }
        },
        async refreshAuth() {
          return tokenManager.refreshToken(async () => {
            const refreshToken = getAuthToken()?.refreshToken
            if (!refreshToken) return undefined

            return utils
              .mutate(refreshTokenMutation, {
                refreshToken
              })
              .then(res => res?.data?.refreshToken)
          })
        }
      }
    }),
    errorExchange({
      onError(error) {
        if (error.message.startsWith('[GraphQL]')) {
          error.message = error.message.replace('[GraphQL]', '').trim()
        }
      }
    }),
    fetchExchange
  ]
})

type QueryVariables<T> = T extends TypedDocumentNode<any, infer U> ? U : never
type QueryResponseData<T> = T extends TypedDocumentNode<infer U, any>
  ? U
  : never

export type {
  ValidationError,
  ValidationErrors,
  QueryVariables,
  QueryResponseData
}
export { client, useMutation }
