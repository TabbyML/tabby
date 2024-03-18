"use client";

import { useState, useEffect } from "react";
import { useRouter } from 'next/navigation'

import { Button } from "@/components/ui/button"
import { UserAuthForm } from './user-register-form'
import AdminRegisterStep from './admin-register-step'

import './admin-register.css'

export default function AdminRegister () {
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState(1)

  useEffect(() => {
    if (currentStep === 1) return
    document.getElementById(`step-${currentStep}`)?.scrollIntoView({ behavior: 'smooth' });
  }, [currentStep])

  return (
    <div className="admin-register-wrap w-[550px]">
      <AdminRegisterStep step={1} currentStep={currentStep}>
        <h2 className="text-3xl font-semibold tracking-tight first:mt-0">
          Welcome To Tabby Enterprise
        </h2>
        <p className="mt-2 leading-7 text-muted-foreground">
          To get started, please create an admin account for your instance.
        </p>
        <p className="leading-7 text-muted-foreground">
          This will allow you to invite team members and manage your instance.
        </p>
        <Button
          className='mt-5 w-48'
          onClick={() => setCurrentStep(2)}>
            Create admin account
        </Button>
      </AdminRegisterStep>

      <AdminRegisterStep step={2} currentStep={currentStep}>
        <h3 className="text-2xl font-semibold tracking-tight">
          Create Admin Account
        </h3>
        <p className="mb-3 leading-7 text-muted-foreground">
          Your instance will be secured, only registered users can access it.
        </p>
        <UserAuthForm onSuccess={() => setCurrentStep(3)} buttonClass="self-start w-48" />
      </AdminRegisterStep>

      <AdminRegisterStep step={3} currentStep={currentStep}>
        <h3 className="text-2xl font-semibold tracking-tight">
          Enter The Instance
        </h3>
        <p className="leading-7 text-muted-foreground">
          Congratulations! You have successfully created an admin account.
        </p>
        <p className="mb-3 leading-7 text-muted-foreground">
        To begin collaborating with your team, please open the dashboard and invite members to join your instance.
        </p>
        <Button
          className='mt-5 w-48'
          onClick={() => router.replace("/")}>
            Open the dashboard
        </Button>
      </AdminRegisterStep>
    </div>
  )
}