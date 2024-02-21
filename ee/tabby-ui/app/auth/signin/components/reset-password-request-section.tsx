'use client'

import React from 'react'
import Link from 'next/link'

import { buttonVariants } from '@/components/ui/button'
import { IconCheckCircled } from '@/components/ui/icons'

import { ResetPasswordRequestForm } from './reset-password-request-form'

export default function ResetPasswordRequestSection() {
  const [email, setEmail] = React.useState<string>()
  const [requestSuccess, setRequestSuccess] = React.useState(false)

  const onSuccess = (email: string) => {
    setEmail(email)
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
            Reset Password
          </h1>
          <p className="pb-4 text-sm text-muted-foreground">
            Request received successfully! If the email{' '}
            <span className="font-bold">{email ?? ''}</span> exists, you’ll
            receive an email with a reset link soon.
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
          Reset Password
        </h1>
        <p className="text-sm text-muted-foreground">
          Enter your email address. If an account exists, you’ll receive an
          email with a password reset link soon.
        </p>
      </div>
      <ResetPasswordRequestForm onSuccess={onSuccess} />
      <div className="text-center">
        <Link
          href="/auth/signin"
          replace
          className="text-primary hover:underline"
        >
          Cancel
        </Link>
      </div>
    </div>
  )
}
