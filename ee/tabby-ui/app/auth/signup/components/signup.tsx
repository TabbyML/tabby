'use client'

import { useSearchParams } from 'next/navigation'

import { UserAuthForm } from './user-register-form'
import AdminRegister from './admin-register'

export default function Signup() {
  const searchParams = useSearchParams()
  const invitationCode = searchParams.get('invitationCode') || undefined
  const isAdmin = searchParams.get('isAdmin') || false
  
  if (isAdmin) return <AdminRegister />
  if (invitationCode) {
    return (
      <Content
        title="Create an account"
        description="Fill form below to create your account"
        show
      />
    )
  }
  return (
    <Content
      title="No invitation code"
      description="Please contact your Tabby admin for an invitation code to register"
    />
  )
}

function Content({
  title,
  description,
  show
}: {
  title: string
  description: string
  show?: boolean
}) {
  const searchParams = useSearchParams()
  const invitationCode = searchParams.get('invitationCode') || undefined

  return (
    <div className="w-[350px] space-y-6">
      <div className="flex flex-col space-y-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      {show && <UserAuthForm invitationCode={invitationCode} />}
    </div>
  )
}
