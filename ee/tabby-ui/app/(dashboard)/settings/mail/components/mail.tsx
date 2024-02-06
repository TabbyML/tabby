'use client'

import { graphql } from '@/lib/gql/generates'
import { useQuery } from 'urql'
import { MailForm } from './mail-form'


const emailSetting = graphql(/* GraphQL */ `
  query emailSetting {
    emailSetting {
      smtpUsername
      smtpServer
    }
  }
`)

export const Mail = () => {

  const [{ data, fetching }] = useQuery({ query: emailSetting })


  return (
    <div>
      <MailForm />
    </div>
  )
}
