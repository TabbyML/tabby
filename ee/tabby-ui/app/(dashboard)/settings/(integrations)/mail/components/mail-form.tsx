'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { useForm, UseFormReturn } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { AuthMethod, Encryption } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger
} from '@/components/ui/alert-dialog'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

const updateEmailSettingMutation = graphql(/* GraphQL */ `
  mutation updateEmailSetting($input: EmailSettingInput!) {
    updateEmailSetting(input: $input)
  }
`)

const deleteEmailSettingMutation = graphql(/* GraphQL */ `
  mutation deleteEmailSetting {
    deleteEmailSetting
  }
`)

const formSchema = z.object({
  smtpUsername: z.string(),
  smtpPassword: z.string(),
  smtpServer: z.string(),
  smtpPort: z.coerce.number({
    invalid_type_error: 'Invalid port',
    required_error: 'Required'
  }),
  fromAddress: z.string(),
  encryption: z.nativeEnum(Encryption),
  authMethod: z.nativeEnum(AuthMethod)
})

type MailFormValues = z.infer<typeof formSchema>

interface MailFormProps {
  isNew?: boolean
  defaultValues?: Partial<MailFormValues>
  onSuccess?: () => void
  onDelete?: () => void
}

interface MailFormRef {
  form: UseFormReturn<MailFormValues>
}

const MailForm = React.forwardRef<MailFormRef, MailFormProps>((props, ref) => {
  const {
    isNew,
    onSuccess,
    onDelete,
    defaultValues: propsDefaultValues
  } = props
  const defaultValues = React.useMemo(() => {
    return {
      encryption: Encryption.None,
      authMethod: AuthMethod.None,
      ...(propsDefaultValues || {})
    }
  }, [propsDefaultValues])

  const form = useForm<MailFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues
  })
  const isDirty = !isEmpty(form.formState.dirtyFields)
  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)

  const updateEmailSetting = useMutation(updateEmailSettingMutation, {
    form,
    onCompleted(data) {
      if (data?.updateEmailSetting) {
        onSuccess?.()
        toast.success('Email configuration is updated.')
      }
    }
  })

  const deleteEmailSetting = useMutation(deleteEmailSettingMutation, {
    onCompleted(data) {
      if (data?.deleteEmailSetting) {
        onDelete?.()
      }
    },
    onError(err) {
      toast.error(err.message)
    }
  })

  const handleDelete: React.MouseEventHandler<HTMLButtonElement> = async e => {
    e.preventDefault()
    await deleteEmailSetting()
  }

  const onSubmit = async (input: MailFormValues) => {
    await updateEmailSetting({
      input: {
        ...input,
        smtpPassword:
          input.smtpPassword !== propsDefaultValues?.smtpPassword
            ? input.smtpPassword
            : undefined
      }
    })
  }

  React.useImperativeHandle(
    ref,
    () => ({
      form
    }),
    [form]
  )

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <div className="flex flex-col gap-6 lg:flex-row">
            <FormField
              control={form.control}
              name="smtpServer"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>SMTP Server Host</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. smtp.gmail.com"
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      className="w-80 min-w-max"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="smtpPort"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>SMTP Server Port</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      placeholder="e.g. 25"
                      className="w-80 min-w-max"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <FormField
            control={form.control}
            name="fromAddress"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>From</FormLabel>
                <FormControl>
                  <Input
                    placeholder="e.g. from@gmail.com"
                    autoCapitalize="none"
                    autoComplete="email"
                    autoCorrect="off"
                    className="w-80 min-w-max"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="authMethod"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Authentication Method</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  name={field.name}
                >
                  <FormControl>
                    <SelectTrigger className="w-80 min-w-max">
                      <SelectValue placeholder="Select a method" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value={AuthMethod.None}>NONE</SelectItem>
                    <SelectItem value={AuthMethod.Plain}>PLAIN</SelectItem>
                    <SelectItem value={AuthMethod.Login}>LOGIN</SelectItem>
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex flex-col gap-6 lg:flex-row">
            <FormField
              control={form.control}
              name="smtpUsername"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>SMTP Username</FormLabel>
                  <FormControl>
                    <Input
                      type="string"
                      placeholder="e.g. support@yourcompany.com"
                      autoCapitalize="none"
                      autoCorrect="off"
                      className="w-80 min-w-max"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="smtpPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>SMTP Password</FormLabel>
                  <FormControl>
                    <Input
                      type="password"
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      className="w-80 min-w-max"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <FormField
            control={form.control}
            name="encryption"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Encryption</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  name={field.name}
                >
                  <FormControl>
                    <SelectTrigger className="w-80 min-w-max">
                      <SelectValue placeholder="Select an encryption" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value={Encryption.None}>NONE</SelectItem>
                    <SelectItem value={Encryption.SslTls}>SSL/TLS</SelectItem>
                    <SelectItem value={Encryption.StartTls}>
                      STARTTLS
                    </SelectItem>
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex items-center gap-4">
            {!isNew && (
              <AlertDialog
                open={deleteAlertVisible}
                onOpenChange={setDeleteAlertVisible}
              >
                <AlertDialogTrigger asChild>
                  <Button variant="hover-destructive">Delete</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>
                      Are you absolutely sure?
                    </AlertDialogTitle>
                    <AlertDialogDescription>
                      This action cannot be undone. It will permanently delete
                      the current setting.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      className={buttonVariants({ variant: 'destructive' })}
                      onClick={handleDelete}
                    >
                      Yes, delete it
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
            <Button type="submit" disabled={!isNew && !isDirty}>
              {isNew ? 'Create' : 'Update'}
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
})

MailForm.displayName = 'MailForm'

export { MailForm }
export type { MailFormRef }
