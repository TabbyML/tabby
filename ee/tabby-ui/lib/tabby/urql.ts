import { authExchange } from '@urql/exchange-auth'
import { cacheExchange, Client, fetchExchange, mapExchange } from 'urql'

import {
  clearAuthData,
  getRefreshToken,
  getToken,
  refreshTokenMutation,
  saveAuthData
} from './auth'

export const client = new Client({
  url: `${process.env.NEXT_PUBLIC_TABBY_SERVER_URL ?? ''}/graphql`,
  exchanges: [
    cacheExchange,
    // mapExchange({
    //   onError(error, _operation) {
    //     const isAuthError = error.graphQLErrors.some(e => e.extensions?.code === 'FORBIDDEN');
    //     if (isAuthError) {
    //       // logout();
    //     }
    //   }
    // }),
    authExchange(async utils => {
      let token = getToken()
      let refreshToken = getRefreshToken()

      return {
        addAuthToOperation(operation) {
          if (!token) return operation
          return utils.appendHeaders(operation, {
            Authorization: `Bearer ${token}`
          })
        },
        didAuthError(error, _operation) {
          // return error.response.status === 401
          return error.graphQLErrors.some(
            e => e.extensions?.code === 'FORBIDDEN'
          )
        },
        willAuthError(operation) {
          // Sync tokens on every operation
          token = getToken()
          refreshToken = getRefreshToken()

          return false
          // Check whether `token` JWT is expired
          if (!token) {
            // Detect our login mutation and let this operation through:

            // todo 1.没有token，2.是mutation，3.不是signin，signup或者refresh token的接口，都返回true，提前失败和re
            return (
              operation.kind !== 'mutation' ||
              // Here we find any mutation definition with the "signin" field
              !operation.query.definitions.some(definition => {
                return (
                  definition.kind === 'OperationDefinition' &&
                  definition.selectionSet.selections.some(node => {
                    // The field name is just an example, since register may also be an exception
                    return node.kind === 'Field' && node.name.value === 'signin'
                  })
                )
              })
            )
          }
          return false

          if (
            operation.kind === 'mutation' &&
            // Here we find any mutation definition with the "login" field
            operation.query.definitions.some(definition => {
              return (
                definition.kind === 'OperationDefinition' &&
                definition.selectionSet.selections.some(node => {
                  // The field name is just an example, since signup may also be an exception
                  return node.kind === 'Field' && node.name.value === 'login'
                })
              )
            })
          ) {
            return false
          } else if (false /* is JWT expired? */) {
            return true
          } else {
            return false
          }
        },
        async refreshAuth() {
          // if not refreshToken, do logout
          if (refreshToken) {
            const result = await utils.mutate(refreshTokenMutation, {
              refreshToken
            })
            if (result.data?.refreshToken) {
              // Update our local variables and write to our storage
              token = result.data.refreshToken.accessToken
              refreshToken = result.data.refreshToken.refreshToken
              saveAuthData({
                token,
                refreshToken
              })
            }
          } else {
            // This is where auth has gone wrong and we need to clean up and redirect to a login page
            // localStorage.clear();
            clearAuthData()
            // send api to singout and redirect to signin page
            // logout();
          }
        }
      }
    }),
    fetchExchange
  ]
})
