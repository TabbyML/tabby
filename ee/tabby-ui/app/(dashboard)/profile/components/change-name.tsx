'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMe } from '@/lib/hooks/use-me'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { ListSkeleton } from '@/components/skeleton'

const updateNameMutation = graphql(/* GraphQL */ `
  mutation UpdateUserName($id: ID!, $name: String!) {
    updateUserName(id: $id, name: $name)
  }
`)

interface ChangeNameFormProps {
  showOldPassword?: boolean
  onSuccess?: () => void
  defaultValues: {
    name?: string
  }
}

const ChangeNameForm: React.FC<ChangeNameFormProps> = ({
  onSuccess,
  defaultValues
}) => {
  const [{ data, fetching }] = useMe()
  const isSsoUser = data?.me?.isSsoUser
  const formSchema = z.object({
    name: z.string()
  })

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues
  })
  const { isSubmitting } = form.formState
  const { name } = form.watch()

  const updateName = useMutation(updateNameMutation, {
    form,
    onCompleted(values) {
      if (values?.updateUserName) {
        onSuccess?.()
      }
    }
  })

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    await updateName({
      id: data!.me.id,
      name: values.name
    })
  }

  const isNameModified = name !== defaultValues.name
  return (
    <Form {...form}>
      <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Input
                      className="w-full md:w-[350px]"
                      {...field}
                      disabled={fetching || isSsoUser}
                    />
                  </TooltipTrigger>
                  <TooltipContent hidden={!isSsoUser} align="end" side="top">
                    The name cannot be altered for users created through SSO
                  </TooltipContent>
                </Tooltip>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormMessage />
        <Separator />
        <div className="flex">
          <Button
            type="submit"
            disabled={!name || !isNameModified || isSubmitting}
            className="w-40"
          >
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Save Changes
          </Button>
        </div>
      </form>
    </Form>
  )
}

export const ChangeName = () => {
  const [{ data }, reexecuteQuery] = useMe()
  const onSuccess = () => {
    toast.success('Name is updated')
    reexecuteQuery()
  }

  return data ? (
    <ChangeNameForm
      onSuccess={onSuccess}
      defaultValues={{ name: data.me.name }}
    />
  ) : (
    <ListSkeleton />
  )
}
