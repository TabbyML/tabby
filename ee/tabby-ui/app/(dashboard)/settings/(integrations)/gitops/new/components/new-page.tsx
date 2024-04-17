'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm, UseFormReturn } from 'react-hook-form'
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
import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { toast } from 'sonner'
import { IconSpinner } from '@/components/ui/icons'

const createGithubRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubRepositoryProvider($displayName: String!, $applicationId: String!, $applicationSecret: String!) {
    createGithubRepositoryProvider(displayName: $displayName, applicationId: $applicationId, applicationSecret: $applicationSecret)
  }
`)

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

  const { isSubmitting } = form.formState

  const createGithubRepositoryProviderMutation = useMutation(createGithubRepositoryProvider, {
    onCompleted(data) {
      if (data?.createGithubRepositoryProvider) {
        toast.success('Created successfully')
        router.push(`/settings/gitops`)
      }
    },
    form
  })

  const handleSubmit = async (
  ) => {
    if (currentStep === 0) {
      // basic info
      setStep(currentStep + 1)
    } else if (currentStep === 1) {
      // oauth application info
      setStep(currentStep + 1)
    } else {
      const values = form.getValues()
      createGithubRepositoryProviderMutation(values)
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
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting && (
                  <IconSpinner className="mr-2" />
                )}
                {currentStep === 2 ? 'Confirm and add' : 'Next'}
              </Button>
            </div>
          </div>
        </form>
      </Form>
    </>
  )
}
