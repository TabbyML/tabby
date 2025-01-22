import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

const meQuery = graphql(/* GraphQL */ `
  query MeQuery {
    me {
      id
      email
      name
      isAdmin
      isOwner
      authToken
      isPasswordSet
      isSsoUser
    }
  }
`)

const useMe = () => {
  return useQuery({ query: meQuery })
}

export { useMe }
