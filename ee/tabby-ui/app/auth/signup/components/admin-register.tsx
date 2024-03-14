"use client";

import { useState } from "react";
import { useRouter } from 'next/navigation'

import { cn } from "@/lib/utils";

import { Button } from "@/components/ui/button"
import { UserAuthForm } from './user-register-form'

import './admin-register.css'

export default function AdminRegister () {
  const router = useRouter()
  const [step, setStep] = useState(1)

  return (
    <div className={cn("admin-register-wrap w-[550px]", {
      "translate-y-1/3": step === 1,
      "-translate-y-10": step === 2,
      "-translate-y-20": step === 3
    })}>

      <div className={cn('border-l border-foreground py-8 pl-12', { 'step-mask': step !== 1, 'remote': step > 2 })}>
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
          onClick={() => setStep(2)}>
            Create admin account
        </Button>
      </div>

      <div className={cn("border-l border-foreground py-8 pl-12", { 'step-mask': step !== 2 })}>
        <h3 className="text-2xl font-semibold tracking-tight">
          Create Admin Account
        </h3>
        <p className="mb-3 leading-7 text-muted-foreground">
          Your instance will be secured, only registered users can access it.
        </p>
        <UserAuthForm onSuccess={() => setStep(3)} buttonClass="self-start w-48" />
      </div>

      <div className={cn("border-l border-foreground py-8 pl-12", { 'step-mask': step !== 3, 'remote': step < 2 })}>
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
      </div>
    </div>
  )
}