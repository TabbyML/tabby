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
import { ListSkeleton } from '@/components/skeleton'

const passwordChangeMutation = graphql(/* GraphQL */ `
  mutation PasswordChange($input: PasswordChangeInput!) {
    passwordChange(input: $input)
  }
`)

interface ChangePasswordFormProps {
  showOldPassword?: boolean
  onSuccess?: () => void
}

const ChangePasswordForm: React.FC<ChangePasswordFormProps> = ({
  onSuccess,
  showOldPassword
}) => {
  const formSchema = z.object({
    oldPassword: showOldPassword ? z.string() : z.string().optional(),
    newPassword1: z.string(),
    newPassword2: z.string()
  })

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  const { isSubmitting } = form.formState

  const passwordChange = useMutation(passwordChangeMutation, {
    form,
    onCompleted(values) {
      if (values?.passwordChange) {
        onSuccess?.()
      }
    }
  })

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    await passwordChange({
      input: values
    })
  }

  return (
    <Form {...form}>
      <div className="flex flex-col gap-4">
        <form
          className="flex flex-col gap-8"
          onSubmit={form.handleSubmit(onSubmit)}
        >
          {showOldPassword && (
            <FormField
              control={form.control}
              name="oldPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Old password</FormLabel>
                  <FormControl>
                    <Input
                      className="w-[350px]"
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      type="password"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          )}
          <FormField
            control={form.control}
            name="newPassword1"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>New password</FormLabel>
                <FormControl>
                  <Input
                    className="w-[350px]"
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    type="password"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="newPassword2"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Confirm new password</FormLabel>
                <FormControl>
                  <Input
                    className="w-[350px]"
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    type="password"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormMessage />
          <Separator />
          <div className="flex">
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Update password
            </Button>
          </div>
        </form>
      </div>
    </Form>
  )
}

export const ChangePassword = () => {
  const [{ data }, reexecuteQuery] = useMe()
  const onSuccess = () => {
    toast.success('Password is updated')
    reexecuteQuery()
  }

  return data ? (
    <ChangePasswordForm
      onSuccess={onSuccess}
      showOldPassword={data?.me?.isPasswordSet}
    />
  ) : (
    <ListSkeleton />
  )
}
