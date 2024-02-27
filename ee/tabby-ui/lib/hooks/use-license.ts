import { useQuery } from 'urql'

import { graphql } from '../gql/generates'

const getLicenseInfo = graphql(/* GraphQL */ `
  query GetLicenseInfo {
    license {
      type
      status
      seats
      seatsUsed
      issuedAt
      expiresAt
    }
  }
`)

const useLicense = () => {
  return useQuery({ query: getLicenseInfo })
}

const useLicenseInfo = () => {
  const [{ data }] = useLicense()
  return data?.license
}

export { getLicenseInfo, useLicense, useLicenseInfo }
