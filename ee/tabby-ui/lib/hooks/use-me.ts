import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

const meQuery = graphql(/* GraphQL */ `
  query MeQuery {
    me {
      authToken
      email
    }
  }
`)

const useMe = () => {
  return useQuery({ query: meQuery })
}

export { useMe }
