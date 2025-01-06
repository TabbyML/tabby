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
  PASSWORD_ERRORCODE,
  PasswordCheckList,
  usePasswordErrors
} from '@/components/password-check-list'
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
  const [showPasswordSchema, setShowPasswordSchema] = React.useState(false)
  const [showPasswordError, setShowPasswordError] = React.useState(false)
  const formSchema = z.object({
    oldPassword: showOldPassword ? z.string() : z.string().optional(),
    newPassword1: z.string(),
    newPassword2: z.string()
  })

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  const { isSubmitting } = form.formState
  const { newPassword1: password } = form.watch()
  const [passworErrors] = usePasswordErrors(password)

  const passwordChange = useMutation(passwordChangeMutation, {
    form,
    onCompleted(values) {
      if (values?.passwordChange) {
        onSuccess?.()
        form.reset({
          newPassword1: '',
          newPassword2: '',
          oldPassword: ''
        })
      }
    }
  })

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    await passwordChange({
      input: values
    })
  }

  const onPasswordBlur = () => {
    if (passworErrors.length === 0) return setShowPasswordSchema(false)
    setShowPasswordError(true)
  }

  return (
    <Form {...form}>
      <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
        {showOldPassword && (
          <FormField
            control={form.control}
            name="oldPassword"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Old password</FormLabel>
                <FormControl>
                  <Input
                    className="w-full md:w-[350px]"
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
        <div>
          <FormField
            control={form.control}
            name="newPassword1"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>New password</FormLabel>
                <FormControl>
                  <Input
                    className="w-full md:w-[350px]"
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    type="password"
                    {...field}
                    onFocus={() => setShowPasswordSchema(true)}
                    onBlur={onPasswordBlur}
                  />
                </FormControl>
              </FormItem>
            )}
          />
          <PasswordCheckList
            password={password || ''}
            showPasswordSchema={showPasswordSchema}
            passworErrors={passworErrors as PASSWORD_ERRORCODE[]}
            showPasswordError={showPasswordError}
          />
        </div>
        <FormField
          control={form.control}
          name="newPassword2"
          render={({ field }) => (
            <FormItem>
              <FormLabel required>Confirm new password</FormLabel>
              <FormControl>
                <Input
                  className="w-full md:w-[350px]"
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
          <Button type="submit" disabled={isSubmitting} className="w-40">
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
