'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { isNil, pickBy } from 'lodash-es'
import { useQuery } from 'urql'

import { ldapCredentialQuery } from '@/lib/tabby/query'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { LDAPCredentialForm } from '../../../components/ldap-credential-form'
import { SSOTypeRadio } from '../../../components/sso-type-radio'

interface OAuthCredentialDetailProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export const LdapCredentialDetail: React.FC<
  OAuthCredentialDetailProps
> = () => {
  const router = useRouter()
  const [{ data, fetching }] = useQuery({
    query: ldapCredentialQuery
  })

  const credential = data?.ldapCredential

  const defaultValues = React.useMemo(() => {
    if (!credential) return undefined
    return pickBy(credential, v => !isNil(v))
  }, [credential])

  const onSubmitSuccess = () => {
    router.push('/settings/sso')
  }

  return (
    <div>
      <LoadingWrapper
        loading={fetching}
        fallback={<ListSkeleton className="mt-2" />}
      >
        <SSOTypeRadio value="ldap" readonly />
        <LDAPCredentialForm
          defaultValues={defaultValues}
          onSuccess={onSubmitSuccess}
          className="mt-6"
        />
      </LoadingWrapper>
    </div>
  )
}
