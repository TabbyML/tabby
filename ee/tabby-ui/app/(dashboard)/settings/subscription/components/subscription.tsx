'use client'

import { capitalize } from 'lodash-es'
import moment from 'moment'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { SubHeader } from '@/components/sub-header'

import { LicenseForm } from './license-form'
import { LicenseTable } from './license-table'

const getLicenseInfo = graphql(/* GraphQL */ `
  query GetLicenseInfo {
    license {
      type
      status
      seats
      issuedAt
      expiresAt
    }
  }
`)

export default function Subscription() {
  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: getLicenseInfo
  })
  const license = data?.license
  const expiresAt = license?.expiresAt
    ? moment(license.expiresAt).format('MM/DD/YYYY')
    : '-'

  const onUploadLicenseSuccess = () => {
    toast.success('License upload successful')
    reexecuteQuery()
  }

  return (
    <div className="p-4">
      <SubHeader>
        You can upload your Tabby license to unlock enterprise features.
      </SubHeader>
      <div className="flex flex-col gap-8">
        <LoadingWrapper
          loading={fetching}
          fallback={
            <div className="grid grid-cols-3 space-x-8">
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
            </div>
          }
        >
          <div className="grid font-bold lg:grid-cols-3">
            <div>
              <div className="mb-1 text-muted-foreground">Current plan</div>
              <div className="text-3xl text-primary">
                {capitalize(license?.type ?? 'FREE')}
              </div>
            </div>
            <div>
              <div className="mb-1 text-muted-foreground">
                Assigned / Total Seats
              </div>
              <div className="text-3xl">0 / {license?.seats ?? '1'}</div>
            </div>
            {!!license && (
              <div>
                <div className="mb-1 text-muted-foreground">Expires at</div>
                <div className="text-3xl">{expiresAt}</div>
              </div>
            )}
          </div>
        </LoadingWrapper>
        <LicenseForm onSuccess={onUploadLicenseSuccess} />
        <LicenseTable />
      </div>
    </div>
  )
}
