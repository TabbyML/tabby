'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

import { OAuthProvider } from '@/lib/gql/generates/graphql'
import { SSOType } from '@/lib/types'

import { LDAPCredentialForm } from '../../components/ldap-credential-form'
import OAuthCredentialForm from '../../components/oauth-credential-form'
import { SSOTypeRadio } from '../../components/sso-type-radio'

export function NewPage() {
  const [type, setType] = useState<SSOType>('oauth')
  const router = useRouter()

  return (
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
  )
}
