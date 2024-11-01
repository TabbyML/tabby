'use client'

import { capitalize } from 'lodash-es'
import moment from 'moment'

import {
  LicenseInfo,
  LicenseStatus,
  LicenseType
} from '@/lib/gql/generates/graphql'
import { useLicense } from '@/lib/hooks/use-license'
import { Badge } from '@/components/ui/badge'
import { IconAlertTriangle } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { SubHeader } from '@/components/sub-header'

import { LicenseForm } from './license-form'
import { LicenseTable } from './license-table'

export default function Subscription() {
  const [{ data, fetching }, reexecuteQuery] = useLicense()
  const license = data?.license
  const onUploadLicenseSuccess = () => {
    reexecuteQuery()
  }
  const canReset = !!license?.type && license.type !== LicenseType.Community

  return (
    <>
      <SubHeader
        className="mb-8"
        externalLink="https://links.tabbyml.com/schedule-a-demo"
        externalLinkText="📆 Book a 30-minute product demo"
      >
        You can upload your Tabby license to unlock team/enterprise features.
      </SubHeader>
      <div className="flex flex-col gap-8">
        <LoadingWrapper
          loading={fetching}
          fallback={
            <div className="grid grid-cols-3">
              <Skeleton className="h-16 w-[80%]" />
              <Skeleton className="h-16 w-[80%]" />
              <Skeleton className="h-16 w-[80%]" />
            </div>
          }
        >
          {license && <License license={license} />}
        </LoadingWrapper>
        <LicenseForm onSuccess={onUploadLicenseSuccess} canReset={canReset} />
        <LicenseTable />
      </div>
    </>
  )
}

function License({ license }: { license: LicenseInfo }) {
  const expiresAt = license.expiresAt
    ? moment(license.expiresAt).format('MM/DD/YYYY')
    : '–'

  const seatsText = `${license.seatsUsed} / ${license.seats}`

  return (
    <div className="grid font-bold lg:grid-cols-3">
      <div>
        <div className="mb-1 text-muted-foreground">Expires at</div>
        <div className="text-3xl flex items-center gap-2">
          {expiresAt}
          {license.status === LicenseStatus.Expired && (
            <Badge variant="destructive" className="flex items-center gap-1">
              <IconAlertTriangle className="h-3 w-3" />
              Expired
            </Badge>
          )}
        </div>
      </div>
      <div>
        <div className="mb-1 text-muted-foreground">Assigned / Total Seats</div>
        <div className="text-3xl flex items-center gap-2">
          {seatsText}
          {license.status === LicenseStatus.Expired && (
            <Badge variant="destructive" className="flex items-center gap-1">
              <IconAlertTriangle className="h-3 w-3" />
              Seats exceeded
            </Badge>
          )}
        </div>
      </div>
      <div>
        <div className="mb-1 text-muted-foreground">Current plan</div>
        <div className="text-3xl text-primary">
          {capitalize(license?.type ?? 'Community')}
        </div>
      </div>
    </div>
  )
}
