'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { compact } from 'lodash-es'
import { useQuery } from 'urql'

import { OAuthProvider } from '@/lib/gql/generates/graphql'
import { ldapCredentialQuery, oauthCredential } from '@/lib/tabby/query'
import { SSOType } from '@/lib/types'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { LDAPCredentialForm } from '../../components/ldap-credential-form'
import OAuthCredentialForm from '../../components/oauth-credential-form'
import { SSOTypeRadio } from '../../components/sso-type-radio'

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

  const fetchingProviders =
    fetchingGithub || fetchingGoogle || fetchingGitlab || fetchingLdap

  const isLdapExisted = !!ldapData?.ldapCredential
  const existedProviders = compact([
    githubData?.oauthCredential && OAuthProvider.Github,
    googleData?.oauthCredential && OAuthProvider.Google,
    gitlabData?.oauthCredential && OAuthProvider.Gitlab
  ])

  const onCreateSuccess = () => {
    router.replace('/settings/sso')
  }

  return (
    <LoadingWrapper loading={fetchingProviders} fallback={<ListSkeleton />}>
      <div>
        <SSOTypeRadio value={type} onChange={setType} />
        {type === 'oauth' ? (
          <OAuthCredentialForm
            isNew
            defaultProvider={OAuthProvider.Github}
            existedProviders={existedProviders}
            onSuccess={onCreateSuccess}
            className="mt-6"
          />
        ) : (
          <LDAPCredentialForm
            isNew
            defaultValues={{ skipTlsVerify: false }}
            onSuccess={onCreateSuccess}
            existed={isLdapExisted}
          />
        )}
      </div>
    </LoadingWrapper>
  )
}
