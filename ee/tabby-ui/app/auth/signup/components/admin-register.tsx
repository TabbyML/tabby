'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

import { UserAuthForm } from './user-register-form'

import './admin-register.css'

function AdminRegisterStep({
  step,
  currentStep,
  children
}: {
  step: number
  currentStep: number
  children: React.ReactNode
}) {
  return (
    <div
      id={`step-${step}`}
      className={cn('border-l border-foreground py-8 pl-12 pr-2', {
        'step-mask': step !== currentStep,
        remote: Math.abs(currentStep - step) > 1
      })}
    >
      {children}
    </div>
  )
}

export default function AdminRegister() {
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState(1)
  const goDashboardBtnRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (currentStep === 1) return
    document
      .getElementById(`step-${currentStep}`)
      ?.scrollIntoView({ behavior: 'smooth' })
  }, [currentStep])

  return (
    <div className="admin-register-wrap h-screen w-[600px] overflow-hidden">
      <AdminRegisterStep step={1} currentStep={currentStep}>
        <h2 className="text-3xl font-semibold tracking-tight first:mt-0">
          Welcome!
        </h2>
        <p className="mt-2 leading-7 text-muted-foreground">
          Your tabby server is live and ready to use. This step by step guide
          will help you set up your admin account.
        </p>
        <p className="leading-7 text-muted-foreground">
          Admin account is the highest level of access in your server. Once
          created, you can invite other members to join your server.
        </p>
        <Button className="mt-5 w-48" onClick={() => setCurrentStep(2)}>
          Start
        </Button>
      </AdminRegisterStep>

      <AdminRegisterStep step={2} currentStep={currentStep}>
        <h3 className="text-2xl font-semibold tracking-tight">
          Create Admin Account
        </h3>
        <p className="mb-3 leading-7 text-muted-foreground">
          Please store your password in a safe place. We do not store your
          password and cannot recover it for you.
        </p>
        <UserAuthForm
          onSuccess={() => {
            setCurrentStep(3)
            goDashboardBtnRef.current?.focus()
          }}
          buttonClass="self-start w-48"
        />
      </AdminRegisterStep>

      <AdminRegisterStep step={3} currentStep={currentStep}>
        <h3 className="text-2xl font-semibold tracking-tight">
          Congratulations!
        </h3>
        <p className="leading-7 text-muted-foreground">
          You have successfully created an admin account.
        </p>
        <p className="mb-3 leading-7 text-muted-foreground">
          To start, navigate to the dashboard and invite other members to join
          your server.
        </p>
        <Button
          className="mt-5 w-48 focus-visible:ring-0"
          onClick={() => router.replace('/')}
          ref={goDashboardBtnRef}
        >
          Go to dashboard
        </Button>
      </AdminRegisterStep>
    </div>
  )
}
