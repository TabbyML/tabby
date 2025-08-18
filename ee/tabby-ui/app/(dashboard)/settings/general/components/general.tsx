'use client'

import React from 'react'

import {
  GetLicenseInfoQuery,
  LicenseFeature,
  LicenseStatus
} from '@/lib/gql/generates/graphql'
import { useLicense } from '@/lib/hooks/use-license'
import { Separator } from '@/components/ui/separator'

import { GeneralBrandingForm as BrandingForm } from './branding-form'
import { GeneralFormSection } from './form-section'
import { GeneralNetworkForm } from './network-form'
import { GeneralSecurityForm } from './security-form'

export default function General() {
  const [{ data }] = useLicense()
  return (
    <div className="flex flex-col gap-4">
      <GeneralFormSection title="Network">
        <GeneralNetworkForm />
      </GeneralFormSection>
      {hasValidLicenseFeature(data, LicenseFeature.CustomLogo) && (
        <>
          <Separator className="mb-8" />
          <GeneralFormSection title="Branding">
            <BrandingForm />
          </GeneralFormSection>
        </>
      )}
      <Separator className="mb-8" />
      <GeneralFormSection title="Security">
        <GeneralSecurityForm />
      </GeneralFormSection>
    </div>
  )
}

const hasValidLicenseFeature = (
  licenseData: GetLicenseInfoQuery | undefined,
  feature: LicenseFeature
): boolean => {
  return (
    !!licenseData?.license &&
    licenseData.license.status === LicenseStatus.Ok &&
    !!licenseData.license.features?.includes(feature)
  )
}
