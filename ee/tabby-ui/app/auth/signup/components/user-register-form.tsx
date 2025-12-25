'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { PLACEHOLDER_EMAIL_FORM } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { useSignIn } from '@/lib/tabby/auth'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
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
import {
  PASSWORD_ERRORCODE,
  PasswordCheckList,
  usePasswordErrors
} from '@/components/password-check-list'

export const registerUser = graphql(/* GraphQL */ `
  mutation register(
    $name: String!
    $email: String!
    $password1: String!
    $password2: String!
    $invitationCode: String
  ) {
    register(
      name: $name
      email: $email
      password1: $password1
      password2: $password2
      invitationCode: $invitationCode
    ) {
      accessToken
      refreshToken
    }
  }
`)

const formSchema = z.object({
  name: z.string(),
  email: z.string().email('Invalid email address'),
  password1: z.string(),
  password2: z.string(),
  invitationCode: z.string().optional()
})

interface UserAuthFormProps extends React.HTMLAttributes<HTMLDivElement> {
  invitationCode?: string
  onSuccess?: () => void
  buttonClass?: string
}

export function UserAuthForm({
  className,
  invitationCode,
  onSuccess,
  buttonClass,
  ...props
}: UserAuthFormProps) {
  const [showPasswordSchema, setShowPasswordSchema] = React.useState(false)
  const [showPasswordError, setShowPasswordError] = React.useState(false)
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      invitationCode
    }
  })

  const { password1: password } = form.watch()
  const [passworErrors] = usePasswordErrors(password)

  const router = useRouter()
  const signIn = useSignIn()
  const { isSubmitting } = form.formState
  const onSubmit = useMutation(registerUser, {
    async onCompleted(values) {
      if (await signIn(values?.register)) {
        if (onSuccess) {
          onSuccess()
        } else {
          router.replace('/')
        }
      }
    },
    form
  })

  const onPasswordBlur = () => {
    if (passworErrors.length === 0) return setShowPasswordSchema(false)
    setShowPasswordError(true)
  }

  return (
    <Form {...form}>
      <div className={cn('grid gap-2', className)} {...props}>
        <form className="grid gap-4" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Name</FormLabel>
                <FormControl>
                  <Input {...field} value={field.value} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input
                    placeholder={`e.g. ${PLACEHOLDER_EMAIL_FORM}`}
                    type="email"
                    autoCapitalize="none"
                    autoComplete="email"
                    autoCorrect="off"
                    {...field}
                    value={field.value ?? ''}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div>
            <FormField
              control={form.control}
              name="password1"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Password</FormLabel>
                  <FormControl>
                    <Input
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
            name="password2"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Confirm Password</FormLabel>
                <FormControl>
                  <Input type="password" {...field} value={field.value ?? ''} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="invitationCode"
            render={({ field }) => (
              <FormItem className="hidden">
                <FormControl>
                  <Input type="hidden" {...field} />
                </FormControl>
              </FormItem>
            )}
          />
          <Button
            type="submit"
            className={cn('mt-2', buttonClass)}
            disabled={isSubmitting}
          >
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Register
          </Button>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
