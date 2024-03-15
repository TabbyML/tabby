'use client'

import React from 'react'
import { OperationResult } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { EmailSettingQuery } from '@/lib/gql/generates/graphql'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { client } from '@/lib/tabby/gql'
import { ListSkeleton } from '@/components/skeleton'

import { MailDeliveryHeader } from './header'
import { MailForm } from './mail-form'
import type { MailFormRef } from './mail-form'
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

const ENCODE_PASSWORD = '********************************'

export const Mail = () => {
  const [queryResult, setQueryResult] =
    React.useState<OperationResult<EmailSettingQuery, any>>()
  const [initialized, setInitialized] = React.useState(false)
  const [debouncedInitialzed] = useDebounceValue(initialized, 200, {
    leading: true
  })
  const mailFormRef = React.useRef<MailFormRef>(null)

  const queryEmailSettings = () => {
    return client
      .query(emailSetting, {})
      .toPromise()
      .then(res => {
        setQueryResult(res)
        setInitialized(true)
        return res
      })
  }

  const isNew = !queryResult?.data?.emailSetting

  const handleMailFormSuccess = () => {
    queryEmailSettings().then(res => {
      const newEmailSettings = res?.data?.emailSetting
      if (newEmailSettings) {
        // reset latest settings
        mailFormRef.current?.form?.reset({
          ...newEmailSettings,
          smtpPassword: ENCODE_PASSWORD
        })
      }
    })
  }

  const handleMailFormDelete = () => {
    // MailForm re-render
    setInitialized(false)
    queryEmailSettings()
  }

  const defaultValues = isNew
    ? {}
    : {
        ...queryResult?.data?.emailSetting,
        smtpPassword: ENCODE_PASSWORD
      }

  React.useEffect(() => {
    queryEmailSettings()
  }, [])

  return (
    <>
      <MailDeliveryHeader />
      {debouncedInitialzed ? (
        <div>
          <div className="mb-8 border-b pb-4">
            <MailForm
              defaultValues={defaultValues}
              isNew={isNew}
              onSuccess={handleMailFormSuccess}
              onDelete={handleMailFormDelete}
              ref={mailFormRef}
            />
          </div>
          <MailTestingForm />
        </div>
      ) : (
        <ListSkeleton />
      )}
    </>
  )
}
