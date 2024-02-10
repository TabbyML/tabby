'use client'

import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { useIsQueryInitialized } from '@/lib/tabby/gql'
import { ListSkeleton } from '@/components/skeleton'

import { MailDeliveryHeader } from './header'
import { MailForm } from './mail-form'
import MailTestingForm from './mail-testing-form'

const emailSetting = graphql(/* GraphQL */ `
  query emailSetting {
    emailSetting {
      smtpUsername
      smtpServer
      fromAddress
      encryption
      authMethod
      smtpPort
    }
  }
`)

export const Mail = () => {
  const [{ data, error }, reexecuteQuery] = useQuery({ query: emailSetting })
  const [initialized] = useIsQueryInitialized({ data, error })

  const isNew = !data?.emailSetting

  const onSendTest = async () => {
    // todo
  }

  const defaultValues = isNew
    ? {}
    : {
        ...data.emailSetting,
        smtpPassword: '********************************'
      }

  return (
    <>
      <MailDeliveryHeader />
      {initialized ? (
        <div>
          <div className="mb-8 border-b pb-4">
            <MailForm
              defaultValues={defaultValues}
              isNew={isNew}
              onSuccess={reexecuteQuery}
              onDelete={reexecuteQuery}
            />
          </div>
          <MailTestingForm onSendTest={onSendTest} />
        </div>
      ) : (
        <ListSkeleton />
      )}
    </>
  )
}
