import { GraphQLClient, Variables, RequestOptions } from 'graphql-request'

export const graphQLClient = new GraphQLClient(
  `${process.env.NEXT_PUBLIC_TABBY_SERVER_URL ?? ''}/graphql`,
  {
    credentials: 'include',
    mode: 'cors'
  }
)

export function request<T, V extends Variables = Variables>(
  options: RequestOptions<V, T>
) {
  return graphQLClient.request(options)
}
