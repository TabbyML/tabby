'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useNetworkSetting } from '@/lib/hooks/use-network-setting'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { ListSkeleton } from '@/components/skeleton'

const updateNetworkSettingMutation = graphql(/* GraphQL */ `
  mutation updateNetworkSettingMutation($input: NetworkSettingInput!) {
    updateNetworkSetting(input: $input)
  }
`)

const formSchema = z.object({
  externalUrl: z.string()
})

type NetworkFormValues = z.infer<typeof formSchema>

interface NetworkFormProps {
  defaultValues?: Partial<NetworkFormValues>
  onSuccess?: () => void
}

const NetworkForm: React.FC<NetworkFormProps> = ({
  onSuccess,
  defaultValues: propsDefaultValues
}) => {
  const defaultValues = React.useMemo(() => {
    return {
      ...(propsDefaultValues || {})
    }
  }, [propsDefaultValues])

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues
  })

  const isDirty = !isEmpty(form.formState.dirtyFields)

  const updateNetworkSetting = useMutation(updateNetworkSettingMutation, {
    form,
    onCompleted(values) {
      if (values?.updateNetworkSetting) {
        onSuccess?.()
        form.reset(form.getValues())
      }
    }
  })

  const onSubmit = async () => {
    await updateNetworkSetting({
      input: form.getValues()
    })
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="externalUrl"
            render={({ field }) => (
              <FormItem>
                <FormLabel>External URL</FormLabel>
                <FormDescription>
                  The external URL where user visits Tabby, must start with
                  http:// or https://.
                </FormDescription>
                <FormControl>
                  <Input
                    placeholder="e.g. http://localhost:8080"
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="mt-2 flex justify-end">
            <Button type="submit" disabled={!isDirty}>
              Update
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}

export const GeneralNetworkForm = () => {
  const [{ data }, reexecuteQuery] = useNetworkSetting({
    requestPolicy: 'network-only'
  })
  const onSuccess = () => {
    toast.success('Network configuration is updated')
    reexecuteQuery()
  }

  return data ? (
    <NetworkForm defaultValues={data.networkSetting} onSuccess={onSuccess} />
  ) : (
    <ListSkeleton />
  )
}
