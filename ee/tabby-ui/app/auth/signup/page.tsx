'use client'

import { UserAuthForm } from './components/user-register-form'
import { useSearchParams } from 'next/navigation'

export default function Signup() {
  const searchParams = useSearchParams()
  const invitationCode = searchParams.get('invitationCode') || undefined
  const isAdmin = searchParams.get("isAdmin") || false;

  const title = isAdmin
    ? 'Create an admin account'
    : 'Create an account';

  return (
    <div className="space-y-6 w-[350px]">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="text-sm text-muted-foreground">
          Fill form below to create your account
        </p>
      </div>
      <UserAuthForm invitationCode={invitationCode} />
    </div>
  )
}
