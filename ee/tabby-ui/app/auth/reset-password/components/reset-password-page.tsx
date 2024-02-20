'use client'

import { useSearchParams } from 'next/navigation'

import { ResetPasswordForm } from './reset-password-form'

export default function ResetPasswordPage() {
  const searchParams = useSearchParams()
  const code = searchParams.get('code') || undefined

  return (
    <div className="w-[350px] space-y-6">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">
          Reset Password
        </h1>
        <p className="text-sm text-muted-foreground">Enter a new password</p>
      </div>
      <ResetPasswordForm code={code} />
    </div>
  )
}
