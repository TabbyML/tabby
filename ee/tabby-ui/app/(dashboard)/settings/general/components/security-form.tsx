'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { compact, isEmpty } from 'lodash-es'
import { useFieldArray, useForm } from 'react-hook-form'
import { toast } from 'sonner'
import { useQuery } from 'urql'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { LicenseType } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconTrash } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { LicenseGuard } from '@/components/license-guard'
import { ListSkeleton } from '@/components/skeleton'

const updateSecuritySettingMutation = graphql(/* GraphQL */ `
  mutation updateSecuritySetting($input: SecuritySettingInput!) {
    updateSecuritySetting(input: $input)
  }
`)

export const securitySetting = graphql(/* GraphQL */ `
  query SecuritySetting {
    securitySetting {
      allowedRegisterDomainList
      disableClientSideTelemetry
    }
  }
`)

const formSchema = z.object({
  disableClientSideTelemetry: z.boolean(),
  // https://github.com/shadcn-ui/ui/issues/384
  // https://github.com/shadcn-ui/ui/blob/main/apps/www/app/examples/forms/profile-form.tsx
  allowedRegisterDomainList: z
    .array(
      z.object({
        value: z.string()
      })
    )
    .optional()
})

type SecurityFormValues = z.infer<typeof formSchema>

interface SecurityFormProps {
  defaultValues?: SecurityFormValues
  onSuccess?: () => void
}

const SecurityForm: React.FC<SecurityFormProps> = ({
  onSuccess,
  defaultValues: propsDefaultValues
}) => {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: propsDefaultValues
  })

  const { fields, append, remove, update } = useFieldArray({
    control: form.control,
    name: 'allowedRegisterDomainList'
  })

  const isDirty = !isEmpty(form.formState.dirtyFields)

  const onRemoveDomainItem = (index: number) => {
    if (fields?.length === 1 && index === 0) {
      update(index, { value: '' })
    } else {
      remove(index)
    }
  }

  const handleDomainListKeyDown: React.KeyboardEventHandler<
    HTMLInputElement
  > = e => {
    if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
      e.preventDefault()
      append({ value: '' })
    }
  }

  const updateSecuritySetting = useMutation(updateSecuritySettingMutation, {
    form,
    onCompleted(values) {
      if (values?.updateSecuritySetting) {
        onSuccess?.()
        form.reset(form.getValues())
      }
    }
  })

  const onSubmit = async ({
    allowedRegisterDomainList,
    ...values
  }: SecurityFormValues) => {
    await updateSecuritySetting({
      input: {
        allowedRegisterDomainList: buildListValuesFromField(
          allowedRegisterDomainList
        ),
        ...values
      }
    })
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="disableClientSideTelemetry"
            render={({ field }) => (
              <FormItem>
                <div className="flex items-center gap-1">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <FormLabel className="cursor-pointer">
                    Disabling Client Side Telemetry
                  </FormLabel>
                </div>
                <FormDescription>
                  When activated, the client-side telemetry (IDE/Extensions)
                  will be disabled, regardless of the client-side settings.
                </FormDescription>
              </FormItem>
            )}
          />
          <div>
            {fields.map((field, index) => (
              <FormField
                control={form.control}
                key={field.id}
                name={`allowedRegisterDomainList.${index}.value`}
                render={({ field: itemField }) => (
                  <FormItem>
                    <FormLabel className={cn(index !== 0 && 'sr-only')}>
                      Authentication Domains
                    </FormLabel>
                    <FormDescription className={cn(index !== 0 && 'sr-only')}>
                      Enable users to sign up automatically with an email
                      address on domains.
                    </FormDescription>
                    <div className="flex items-center gap-2">
                      <FormControl>
                        <Input
                          placeholder="e.g. tabbyml.com"
                          {...itemField}
                          onKeyDown={handleDomainListKeyDown}
                        />
                      </FormControl>
                      <Button
                        variant="hover-destructive"
                        onClick={e => onRemoveDomainItem(index)}
                      >
                        <IconTrash />
                      </Button>
                    </div>
                    <FormMessage />
                  </FormItem>
                )}
              />
            ))}
            <div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="mt-2"
                onClick={() => append({ value: '' })}
              >
                Add domain
              </Button>
            </div>
          </div>
          <div className="mt-2 flex justify-end">
            <LicenseGuard licenses={[LicenseType.Enterprise]}>
              {({ hasValidLicense }) => {
                return (
                  <Button type="submit" disabled={!hasValidLicense || !isDirty}>
                    Update
                  </Button>
                )
              }}
            </LicenseGuard>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}

function buildListFieldFromValues(list: string[] | undefined) {
  const domains = list?.map(item => ({ value: item }))
  if (!domains || domains.length === 0) {
    return [{ value: '' }]
  } else {
    return domains
  }
}

function buildListValuesFromField(fieldListValue?: Array<{ value: string }>) {
  const list = compact(fieldListValue?.map(item => item.value))
  return list
}

export const GeneralSecurityForm = () => {
  const [{ data }, reexecuteQuery] = useQuery({
    query: securitySetting,
    requestPolicy: 'network-only'
  })
  const onSuccess = () => {
    toast.success('Security configuration is updated')
    reexecuteQuery()
  }
  const defaultValues = data && {
    ...data.securitySetting,
    allowedRegisterDomainList: buildListFieldFromValues(
      data.securitySetting.allowedRegisterDomainList
    )
  }
  return data ? (
    <SecurityForm defaultValues={defaultValues} onSuccess={onSuccess} />
  ) : (
    <ListSkeleton />
  )
}
