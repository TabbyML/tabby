'use client'

import { useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { createRequest } from '@urql/core'
import { omit } from 'lodash-es'
import { useForm, UseFormReturn } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { usePopupWindow } from '@/lib/popup-window-management'
import { client, useMutation } from '@/lib/tabby/gql'
import { listGithubRepositoryProviders } from '@/lib/tabby/query'
import { getAuthToken } from '@/lib/tabby/token-management'
import { Button } from '@/components/ui/button'
import { Form } from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
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

const createGithubRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubRepositoryProvider(
    $input: CreateGithubRepositoryProviderInput!
  ) {
    createGithubRepositoryProvider(input: $input)
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
  const [errorMessage, setErrorMessage] = useState<string | undefined>()

  const createdProviderId = useRef<string | undefined>()
  const { isSubmitting } = form.formState

  const getProvider = (id: string) => {
    const queryProvider = client.createRequestOperation(
      'query',
      createRequest(listGithubRepositoryProviders, { ids: [id] })
    )
    return client.executeQuery(queryProvider)
  }

  const getPopupUrl = (id: string) => {
    const accessToken = getAuthToken()?.accessToken
    return `/integrations/github/connect/${id}?access_token=${accessToken}`
  }

  const { open: openPopup } = usePopupWindow({
    async onMessage(data) {
      if (data?.errorMessage) {
        setErrorMessage(data.errorMessage)
      } else {
        const result = await getProvider(createdProviderId.current as string)
        if (
          result?.data?.githubRepositoryProviders?.edges?.[0]?.node?.connected
        ) {
          toast.success('Provider Successfully Created')
          router.replace('/settings/gitops')
        } else {
          setErrorMessage('Connection to GitHub failed, please try again')
        }
      }
    }
  })

  const createGithubRepositoryProviderMutation = useMutation(
    createGithubRepositoryProvider,
    {
      onCompleted(data) {
        if (data?.createGithubRepositoryProvider) {
          // store providerId
          createdProviderId.current = data.createGithubRepositoryProvider
          openPopup(getPopupUrl(data.createGithubRepositoryProvider))
          setErrorMessage(undefined)
        }
      },
      onError(err) {
        toast.error(err?.message)
      },
      form
    }
  )

  const handleSubmit = async () => {
    if (currentStep === 0) {
      // basic info
      setStep(currentStep + 1)
    } else if (currentStep === 1) {
      // oauth application info
      setStep(currentStep + 1)
    } else {
      if (createdProviderId.current) {
        openPopup(getPopupUrl(createdProviderId.current))
        setErrorMessage(undefined)
        return
      }

      const values = omit(form.getValues(), 'provider')
      createGithubRepositoryProviderMutation({
        input: values
      })
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
          {currentStep === 2 && (
            <>
              <ConfirmView data={form.getValues()} />
              <div className="mt-2 text-center text-destructive-foreground">
                {errorMessage}
              </div>
            </>
          )}
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
                {isSubmitting && <IconSpinner className="mr-2" />}
                {currentStep === 2 ? 'Confirm and connect' : 'Next'}
              </Button>
            </div>
          </div>
        </form>
      </Form>
    </>
  )
}
