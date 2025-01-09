'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

import { OAuthProvider } from '@/lib/gql/generates/graphql'
import { SSOType } from '@/lib/types'

import { LDAPCredentialForm } from '../../components/ldap-credential-form'
import OAuthCredentialForm from '../../components/oauth-credential-form'
import { SSOTypeRadio } from '../../components/sso-type-radio'
import { useQuery } from 'urql'
import { ldapCredentialQuery, oauthCredential } from '@/lib/tabby/query'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'
import { compact } from 'lodash-es'

export function NewPage() {
  const [type, setType] = useState<SSOType>('oauth')
  const router = useRouter()
  const [{ data: githubData, fetching: fetchingGithub }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Github }
  })
  const [{ data: googleData, fetching: fetchingGoogle }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Google }
  })
  const [{ data: gitlabData, fetching: fetchingGitlab }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Gitlab }
  })

  const [{ data: ldapData, fetching: fetchingLdap }] = useQuery({
    query: ldapCredentialQuery
  })

  const fetching = fetchingGithub || fetchingGoogle || fetchingGitlab || fetchingLdap
  const isOauthAvaliable = compact([githubData?.oauthCredential, googleData?.oauthCredential, gitlabData?.oauthCredential]).length < 3
  const isLdapAvaliable = !ldapData?.ldapCredential

  const onCreateSuccess = () => {
    router.replace('/settings/sso')
  }

  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<ListSkeleton />}
    >
      <div>
        <SSOTypeRadio value={type} onChange={setType} />
        {type === 'oauth' ? (
          <OAuthCredentialForm provider={OAuthProvider.Github} isNew />
        ) : (
          <LDAPCredentialForm
            isNew
            defaultValues={{ skipTlsVerify: false }}
            onSuccess={() => {
              router.replace('/settings/sso')
            }}
          />
        )}
      </div>
    </LoadingWrapper>
  )
}
