'use client'

import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { useIsQueryInitialized } from '@/lib/tabby/gql'

import { MailDeliveryHeader } from './header'
import { MailForm } from './mail-form'
import MailTestingForm from './mail-testing-form'

const emailSetting = graphql(/* GraphQL */ `
  query emailSetting {
    emailSetting {
      smtpUsername
      smtpServer
    }
  }
`)

export const Mail = () => {
  const [{ data, error }] = useQuery({ query: emailSetting })
  const [initialized] = useIsQueryInitialized({ data, error })

  const isNew = !data?.emailSetting

  const onSendTest = async () => {}

  return (
    <>
      <MailDeliveryHeader />
      {initialized ? (
        <div>
          <div className="pb-4 mb-8 border-b">
            <MailForm isNew={isNew} />
          </div>
          <MailTestingForm onSendTest={onSendTest} />
        </div>
      ) : null}
    </>
  )
}
