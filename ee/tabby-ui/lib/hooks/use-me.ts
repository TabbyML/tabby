import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

const meQuery = graphql(/* GraphQL */ `
  query MeQuery {
    me {
      id
      authToken
      email
      isAdmin
      isOwner
      isPasswordSet
      name
    }
  }
`)

const useMe = () => {
  return useQuery({ query: meQuery })
}

export { useMe }
