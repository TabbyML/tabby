import { TypedDocumentNode } from '@graphql-typed-document-node/core'
import { authExchange } from '@urql/exchange-auth'
import { cacheExchange } from '@urql/exchange-graphcache'
import { relayPagination } from '@urql/exchange-graphcache/extras'
import { createClient as createWSClient } from 'graphql-ws'
import { jwtDecode } from 'jwt-decode'
import { isNil } from 'lodash-es'
import { FieldValues, UseFormReturn } from 'react-hook-form'
import {
  AnyVariables,
  Client,
  CombinedError,
  errorExchange,
  fetchExchange,
  OperationContext,
  OperationResult,
  subscriptionExchange,
  useMutation as useUrqlMutation
} from 'urql'

import {
  DeleteUserGroupMembershipMutationVariables,
  GitRepositoriesQueryVariables,
  ListIntegrationsQueryVariables,
  ListInvitationsQueryVariables,
  ListPageSectionsQueryVariables,
  ListThreadsQueryVariables,
  NotificationsQueryVariables,
  SourceIdAccessPoliciesQueryVariables,
  UpsertUserGroupMembershipInput
} from '../gql/generates/graphql'
import { ExtendedCombinedError } from '../types'
import { refreshTokenMutation } from './auth'
import {
  listIntegrations,
  listInvitations,
  listPageSections,
  listRepositories,
  listSourceIdAccessPolicies,
  listThreads,
  notificationsQuery,
  userGroupsQuery
} from './query'
import {
  getAuthToken,
  getFetcherOptions,
  isTokenExpired,
  tokenManager
} from './token-management'

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

  const fn = async (
    variables?: TVariables & { extraParams?: any },
    context?: Partial<OperationContext>
  ) => {
    let response: OperationResult<TResult, AnyVariables> | undefined

    try {
      response = await executeMutation(variables, context)
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

export function makeFormErrorHandler<T extends FieldValues>(
  form: UseFormReturn<T>
) {
  return (err: ExtendedCombinedError) => {
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
      } else if (error?.message) {
        form.setError('root', { message: error.message })
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
        RepositoryList: () => null,
        RepositoryGrep: () => null,
        GrepLine: () => null,
        GrepFile: () => null,
        GrepTextOrBase64: () => null,
        GrepSubMatch: () => null,
        GitReference: () => null,
        MessageAttachment: () => null,
        MessageAttachmentCode: () => null,
        MessageAttachmentDoc: () => null,
        NetworkSetting: () => null,
        ContextInfo: () => null
      },
      resolvers: {
        Query: {
          invitations: relayPagination(),
          gitRepositories: relayPagination(),
          webCrawlerUrls: relayPagination(),
          integrations: relayPagination(),
          threads: relayPagination()
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
          },
          deleteIntegration(result, args, cache, info) {
            if (result.deleteIntegration) {
              cache
                .inspectFields('Query')
                .filter(field => field.fieldName === 'integrations')
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: listIntegrations,
                      variables:
                        field.arguments as ListIntegrationsQueryVariables
                    },
                    data => {
                      if (data?.integrations) {
                        data.integrations.edges =
                          data.integrations.edges.filter(
                            e => e.node.id !== args.id
                          )
                      }
                      return data
                    }
                  )
                })
            }
          },
          createIntegration(result, args, cache) {
            const key = 'Query'
            cache
              .inspectFields(key)
              .filter(field => {
                return (
                  field.fieldName === 'integrations' &&
                  !!field.arguments?.kind &&
                  // @ts-ignore
                  field.arguments?.kind === args?.input?.kind
                )
              })
              .forEach(field => {
                cache.invalidate(key, field.fieldName, field.arguments)
              })
          },
          upsertUserGroupMembership(result, args, cache, info) {
            const { userGroupId, userId, isGroupAdmin } =
              args.input as UpsertUserGroupMembershipInput
            const { user, isInsert } = (info.variables.extraParams || {}) as any
            if (result.upsertUserGroupMembership) {
              cache.updateQuery({ query: userGroupsQuery }, data => {
                if (data?.userGroups) {
                  data.userGroups = data.userGroups.map(group => {
                    if (group.id !== userGroupId) return group
                    let newMembers = [...group.members]
                    if (isInsert) {
                      const now = new Date().toISOString()
                      newMembers.push({
                        user: {
                          ...user,
                          __typename: 'UserSecured'
                        },
                        isGroupAdmin,
                        createdAt: now,
                        updatedAt: now,
                        __typename: 'UserGroupMembership'
                      })
                    } else {
                      newMembers = newMembers.map(m => {
                        if (m.user.id !== userId) return m
                        return {
                          ...m,
                          isGroupAdmin
                        }
                      })
                    }
                    return {
                      ...group,
                      members: newMembers
                    }
                  })
                }

                return data
              })
            }
          },
          deleteUserGroupMembership(result, args, cache, info) {
            const { userGroupId, userId } =
              args as DeleteUserGroupMembershipMutationVariables
            if (result.deleteUserGroupMembership) {
              cache.updateQuery({ query: userGroupsQuery }, data => {
                if (data?.userGroups) {
                  data.userGroups = data.userGroups.map(group => {
                    if (group.id !== userGroupId) return group
                    let newMembers = [...group.members].filter(
                      o => o.user.id !== userId
                    )
                    return {
                      ...group,
                      members: newMembers
                    }
                  })
                }
                return data
              })
            }
          },
          grantSourceIdReadAccess(
            result,
            args: { sourceId: string; userGroupId: string },
            cache,
            info
          ) {
            if (result.grantSourceIdReadAccess) {
              const { sourceId } = args
              cache
                .inspectFields('Query')
                .filter(
                  field =>
                    field.fieldName === 'sourceIdAccessPolicies' &&
                    field.arguments?.sourceId === sourceId
                )
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: listSourceIdAccessPolicies,
                      variables:
                        field.arguments as SourceIdAccessPoliciesQueryVariables
                    },
                    data => {
                      if (data?.sourceIdAccessPolicies?.read) {
                        const { userGroupName } = (info.variables.extraParams ||
                          {}) as any
                        data.sourceIdAccessPolicies.read = [
                          ...data.sourceIdAccessPolicies.read,
                          {
                            __typename: 'UserGroup',
                            id: args.userGroupId,
                            name: userGroupName
                          }
                        ]
                      }
                      return data
                    }
                  )
                })
            }
          },
          revokeSourceIdReadAccess(
            result,
            args: { sourceId: string; userGroupId: string },
            cache,
            info
          ) {
            if (result.revokeSourceIdReadAccess) {
              const { userGroupId, sourceId } = args
              cache
                .inspectFields('Query')
                .filter(
                  field =>
                    field.fieldName === 'sourceIdAccessPolicies' &&
                    field.arguments?.sourceId === sourceId
                )
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: listSourceIdAccessPolicies,
                      variables:
                        field.arguments as SourceIdAccessPoliciesQueryVariables
                    },
                    data => {
                      if (
                        data?.sourceIdAccessPolicies?.sourceId === sourceId &&
                        data?.sourceIdAccessPolicies?.read
                      ) {
                        data.sourceIdAccessPolicies.read =
                          data.sourceIdAccessPolicies.read.filter(
                            o => o.id !== userGroupId
                          )
                      }
                      return data
                    }
                  )
                })
            }
          },
          deleteThread(result, args, cache, info) {
            if (result.deleteThread) {
              cache
                .inspectFields('Query')
                // Update the cache within the thread-feeds only
                .filter(
                  field =>
                    field.fieldName === 'threads' && !field.arguments?.ids
                )
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: listThreads,
                      variables: field.arguments as ListThreadsQueryVariables
                    },
                    data => {
                      if (data?.threads) {
                        data.threads.edges = data.threads.edges.filter(
                          e => e.node.id !== args.id
                        )
                      }
                      return data
                    }
                  )
                })
            }
          },
          setThreadPersisted(result, args, cache, info) {
            if (result.setThreadPersisted) {
              const key = 'Query'
              cache
                .inspectFields(key)
                .filter(field => {
                  return (
                    field.fieldName === 'threads' &&
                    !field.arguments?.ids &&
                    !!field.arguments?.before
                  )
                })
                .forEach(field => {
                  cache.invalidate(key, field.fieldName, field.arguments)
                })
            }
          },
          markNotificationsRead(result, args, cache) {
            if (result.markNotificationsRead) {
              cache
                .inspectFields('Query')
                .filter(field => field.fieldName === 'notifications')
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: notificationsQuery,
                      variables: field.arguments as NotificationsQueryVariables
                    },
                    data => {
                      if (data?.notifications) {
                        const isMarkAllAsRead = !args.notificationId
                        data.notifications = data.notifications.map(item => {
                          if (isMarkAllAsRead) {
                            return {
                              ...item,
                              read: true
                            }
                          } else {
                            if (item.id === args.notificationId) {
                              return {
                                ...item,
                                read: true
                              }
                            }
                            return item
                          }
                        })
                      }
                      return data
                    }
                  )
                })
            }
          },
          deletePageSection(result, args, cache) {
            if (result.deletePageSection) {
              cache
                .inspectFields('Query')
                .filter(field => field.fieldName === 'pageSections')
                .forEach(field => {
                  cache.updateQuery(
                    {
                      query: listPageSections,
                      variables:
                        field.arguments as ListPageSectionsQueryVariables
                    },
                    data => {
                      if (data?.pageSections) {
                        data.pageSections.edges =
                          data.pageSections.edges.filter(
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
      },
      optimistic: {
        upsertUserGroupMembership() {
          return true
        },
        deleteUserGroupMembership() {
          return true
        },
        grantSourceIdReadAccess() {
          return true
        },
        revokeSourceIdReadAccess() {
          return true
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
          const fetcherOptions = getFetcherOptions()
          accessToken = authData?.accessToken
          refreshToken = authData?.refreshToken

          if (accessToken) {
            return utils.appendHeaders(operation, {
              Authorization: `Bearer ${accessToken}`
            })
          } else if (fetcherOptions) {
            const headers = {
              Authorization: `Bearer ${fetcherOptions.authorization}`,
              ...fetcherOptions.headers
            }
            return utils.appendHeaders(operation, headers)
          }

          return operation
        },
        didAuthError(error, _operation) {
          const isUnauthorized = error.graphQLErrors.some(
            e => e?.extensions?.code === 'UNAUTHORIZED'
          )
          if (isUnauthorized) {
            tokenManager.clearAccessToken()
          }

          return isUnauthorized
        },
        willAuthError(operation) {
          // Sync tokens on every operation
          const authData = getAuthToken()
          const fetcherOptions = getFetcherOptions()
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
          } else if (fetcherOptions) {
            return !fetcherOptions?.authorization
          } else {
            tokenManager.clearAccessToken()
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
    fetchExchange,
    subscriptionExchange({
      forwardSubscription(request, operation) {
        const authorization =
          // @ts-ignore
          operation.context.fetchOptions?.headers?.Authorization ?? ''
        const protocol = window.location.protocol
        const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        const wsClient = createWSClient({
          url: `${wsProtocol}//${host}/subscriptions`,
          connectionParams: {
            authorization
          }
        })
        const input = { ...request, query: request.query || '' }
        return {
          subscribe(sink) {
            const unsubscribe = wsClient.subscribe(input, sink)
            return { unsubscribe }
          }
        }
      }
    })
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
