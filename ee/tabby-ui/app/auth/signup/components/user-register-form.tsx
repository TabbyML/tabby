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

export const registerUser = graphql(/* GraphQL */ `
  mutation register(
    $email: String!
    $password1: String!
    $password2: String!
    $invitationCode: String
  ) {
    register(
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

enum PASSWORD_ERRORCODE {
  LOWERCASE_MSISSING = 'lowercase_missing',
  UPPERCASE_MSISSING = 'uppercase_missing',
  NUMBER_MISSING = 'number_missing',
  SPECIAL_CHAR_MISSING ='special_char_missing',
}

const passwordSchema = z.string()
  .refine((password) => /[a-z]/.test(password), {
    message: "Password should contain at least one lowercase character",
    params: { errorCode: PASSWORD_ERRORCODE.LOWERCASE_MSISSING }
  })
  .refine((password) => /[A-Z]/.test(password), {
    message: "Password should contain at least one uppercase character",
    params: { errorCode: PASSWORD_ERRORCODE.UPPERCASE_MSISSING }
  })
  .refine((password) => /\d/.test(password), {
    message: "Password should contain at least one numeric character",
    params: { errorCode: PASSWORD_ERRORCODE.NUMBER_MISSING }
  })
  .refine((password) => /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]+/.test(password), {
    message: "Password should contain at least one special character, e.g @#$%^&{}",
    params: { errorCode: PASSWORD_ERRORCODE.SPECIAL_CHAR_MISSING }
  });

const formSchema = z.object({
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
  const [password, setPassword] = React.useState("")
  const [showPasswordSchema, setShowPasswordSchema] = React.useState(false)
  const [passworErrors, setPasswordErrors] = React.useState<PASSWORD_ERRORCODE[]>([])
  const [showPasswordError, setShowPasswordError] = React.useState(false)
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      invitationCode
    }
  })
 
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

  const onChangePassword = (e: React.ChangeEvent<HTMLInputElement>) => {
    const password = e.target.value
    form.setValue('password1', password)
    setPassword(password)
    try {
      passwordSchema.parse(password)
      setPasswordErrors([])
    } catch (err) {
      if (err instanceof z.ZodError) {
        setPasswordErrors(err.issues.map((error: any) => error.params.errorCode))
      }
    }
  }

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
                    value={password}
                    onChange={onChangePassword}
                    onFocus={() => setShowPasswordSchema(true)}
                    onBlur={onPasswordBlur} />
                </FormControl>
              </FormItem>
            )}
          />
          <div className={cn("relative text-sm transition-all", {
            'h-0 opacity-0 -z-10': !showPasswordSchema,
            'h-28 opacity-100': showPasswordSchema
          })}>
            <p className="mb-0.5 text-xs text-muted-foreground">Set up a strong password with at least</p>
            <ul className="list-disc pl-4">
              <li className={cn("py-0.5", { 'text-green-600': password.length > 0 && !passworErrors.includes(PASSWORD_ERRORCODE.LOWERCASE_MSISSING), 'text-red-600': showPasswordError && password.length > 0 && passworErrors.includes(PASSWORD_ERRORCODE.LOWERCASE_MSISSING) })}>One lowercase character</li>
              <li className={cn("py-0.5", { 'text-green-600': password.length > 0 && !passworErrors.includes(PASSWORD_ERRORCODE.UPPERCASE_MSISSING), 'text-red-600': showPasswordError && password.length > 0 && passworErrors.includes(PASSWORD_ERRORCODE.UPPERCASE_MSISSING) })}>One uppercase character</li>
              <li className={cn("py-0.5", { 'text-green-600': password.length > 0 && !passworErrors.includes(PASSWORD_ERRORCODE.NUMBER_MISSING), 'text-red-600': showPasswordError && password.length > 0 && passworErrors.includes(PASSWORD_ERRORCODE.NUMBER_MISSING) })}>One numeric character</li>
              <li className={cn("py-0.5", { 'text-green-600': password.length > 0 && !passworErrors.includes(PASSWORD_ERRORCODE.SPECIAL_CHAR_MISSING), 'text-red-600': showPasswordError && password.length > 0 && passworErrors.includes(PASSWORD_ERRORCODE.SPECIAL_CHAR_MISSING) })}>{`One special character, such as @#$%^&{}`}</li>
            </ul>
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
