import { useSearchParams } from 'next/navigation'
import { useQuery } from 'urql'

import { graphql } from '../gql/generates'
import { LicenseStatus, LicenseType } from '../gql/generates/graphql'

const getLicenseInfo = graphql(/* GraphQL */ `
  query GetLicenseInfo {
    license {
      type
      status
      seats
      seatsUsed
      issuedAt
      expiresAt
      features
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

type UseLicenseValidityOptions = {
  /**
   * requiredLicenses
   */
  licenses?: LicenseType[]
  /**
   * Indicates whether the operation permissions for the child component
   * are related to the seat count.
   *
   * If `true`, the component will check if the seat count exceeds the limit
   * before allowing the operation.
   */
  // isSeatCountRelated?: boolean
}

export type UseLicenseValidityResponse = ReturnType<typeof useLicenseValidity>

/**
 * check the validity of the current license
 */
export const useLicenseValidity = (options?: UseLicenseValidityOptions) => {
  const [{ data }] = useLicense()
  const licenseInfo = data?.license
  const searchParams = useSearchParams()

  const hasLicense = !!licenseInfo

  // Determine if the current license has sufficient level
  const hasSufficientLicense =
    !!licenseInfo &&
    (!options?.licenses?.length || options.licenses.includes(licenseInfo.type))

  const isLicenseOK = licenseInfo?.status === LicenseStatus.Ok

  // Determine if the current license is expired
  const isExpired = licenseInfo?.status === LicenseStatus.Expired

  // Determine if the seat count is exceeded
  const isSeatsExceeded = licenseInfo?.status === LicenseStatus?.SeatsExceeded

  // Testing parameters from searchParams
  const isTestExpired = searchParams.get('licenseError') === 'expired'
  const isTestSeatsExceeded = searchParams.get('licenseError') === 'seatsExceed'

  return {
    hasLicense,
    // FIXME introduced testing searchParams
    isLicenseOK: isLicenseOK && !(isTestExpired || isTestSeatsExceeded),
    isExpired: isExpired || isTestExpired,
    isSeatsExceeded: isSeatsExceeded || isTestSeatsExceeded,
    hasSufficientLicense
  }
}

export { getLicenseInfo, useLicense, useLicenseInfo }
