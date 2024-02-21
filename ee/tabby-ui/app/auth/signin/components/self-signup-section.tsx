'use client'

import React from 'react'
import Link from 'next/link'

import { RequestInvitationEmailMutation } from '@/lib/gql/generates/graphql'
import { buttonVariants } from '@/components/ui/button'
import { IconCheckCircled } from '@/components/ui/icons'

import { SelfSignupForm } from './self-signup-form'

export default function SelfSignupSection() {
  const [email, setEmail] = React.useState<string>()
  const [requestSuccess, setRequestSuccess] = React.useState(false)

  const onSuccess = (
    data: RequestInvitationEmailMutation['requestInvitationEmail']
  ) => {
    setEmail(data.email)
    setRequestSuccess(true)
  }

  if (requestSuccess) {
    return (
      <div className="w-[350px] space-y-6">
        <div className="flex flex-col space-y-2 text-center">
          <div className="flex justify-center">
            <IconCheckCircled className="h-12 w-12 text-successful-foreground" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Create your Tabby account
          </h1>
          <p className="pb-4 text-sm text-muted-foreground">
            Request received successfully! You’ll receive an email with a signup
            link soon.
          </p>
          <Link href="/auth/signin" className={buttonVariants()}>
            Back to Sign In
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="w-[350px] space-y-6">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">
          Create your Tabby account
        </h1>
        <p className="text-sm text-muted-foreground">
          To register your account, please enter your email address.
        </p>
      </div>
      <SelfSignupForm onSuccess={onSuccess} />
      <div className="text-center text-sm">
        Already have an accout?
        <Link
          href="/auth/signin"
          className="ml-1 font-semibold text-primary hover:underline"
        >
          Sign In
        </Link>
      </div>
    </div>
  )
}
