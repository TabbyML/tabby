'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { uniqueId } from 'lodash-es'
import { useForm, UseFormReturn } from 'react-hook-form'
import useLocalStorage from 'use-local-storage'
import * as z from 'zod'

import { Button } from '@/components/ui/button'
import { Form } from '@/components/ui/form'
import { StepItem, Steps, useSteps } from '@/components/steps/steps'

import {
  BasicInfoForm,
  basicInfoFormSchema,
  type BasicInfoFormValues
} from '../../components/basic-info-form'
import { RepositoryHeader } from '../../components/header'
import {
  OAuthApplicationForm,
  oauthInfoFormSchema,
  type OAuthApplicationFormValues
} from '../../components/oauth-application-form'
import ConfirmView from '../components/confirm-view'

export const NewProvider = () => {
  const router = useRouter()
  const schemas = [basicInfoFormSchema, oauthInfoFormSchema, z.object({})]

  const stepState = useSteps({
    items: [
      {
        title: 'Basic info'
      },
      {
        title: 'OAuth application info'
      },
      {
        title: 'Confirmation'
      }
    ]
  })
  const { currentStep, setStep } = stepState
  const form = useForm<BasicInfoFormValues & OAuthApplicationFormValues>({
    resolver: zodResolver(schemas[currentStep]),
    defaultValues: {
      provider: 'github'
    }
  })

  // todo mock, remove later
  const [mockGitopsData, setMockGitopsData] = useLocalStorage<Array<
    (BasicInfoFormValues | OAuthApplicationFormValues) & { id: string }
  > | null>('mock-gitops-data', null)

  const handleSubmit = (
    values: BasicInfoFormValues | OAuthApplicationFormValues
  ) => {
    if (currentStep === 0) {
      // basic info
      setStep(currentStep + 1)
    } else if (currentStep === 1) {
      // oauth application info
      setStep(currentStep + 1)
    } else {
      const currentMockGitopsData = mockGitopsData || []
      const allValues = form.getValues()
      // mock
      setMockGitopsData([
        ...currentMockGitopsData,
        { id: uniqueId(), ...allValues }
      ])
      router.push('/settings/gitops')
    }
  }

  return (
    <>
      <RepositoryHeader />
      <Steps {...stepState}>
        {stepState.steps?.map((step, index) => {
          return <StepItem key={index} index={index} {...step} />
        })}
      </Steps>
      <Form {...form}>
        <form className="mt-6" onSubmit={form.handleSubmit(handleSubmit)}>
          {currentStep === 0 && (
            <BasicInfoForm form={form as UseFormReturn<any>} />
          )}
          {currentStep === 1 && (
            <OAuthApplicationForm form={form as UseFormReturn<any>} />
          )}
          {currentStep === 2 && <ConfirmView data={form.getValues()} />}
          <div className="mt-8 flex justify-between">
            <Button
              type="button"
              variant="ghost"
              onClick={e => router.push('/settings/gitops')}
            >
              Cancel
            </Button>
            <div className="flex gap-4">
              {currentStep > 0 && (
                <Button
                  variant="secondary"
                  type="button"
                  onClick={e => setStep(currentStep - 1)}
                >
                  Back
                </Button>
              )}
              <Button type="submit">
                {currentStep === 2 ? 'Confirm and add' : 'Next'}
              </Button>
            </div>
          </div>
        </form>
      </Form>
    </>
  )
}
