'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
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
import { IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

const updateEmailSettingMutation = graphql(/* GraphQL */ `
  mutation updateEmailSetting(
    $smtpUsername: String!
    $smtpPassword: String
    $smtpServer: String!
  ) {
    updateEmailSetting(
      smtpUsername: $smtpUsername
      smtpPassword: $smtpPassword
      smtpServer: $smtpServer
    )
  }
`)

const deleteEmailSettingMutation = graphql(/* GraphQL */ `
  mutation deleteEmailSetting {
    deleteEmailSetting
  }
`)

interface MailFormProps {
  isNew?: boolean
  defaultValues?: any
  onSuccess?: () => void
  onDelete?: () => void
}

const formSchema = z.object({
  smtpUsername: z.string(),
  smtpPassword: z.string(),
  smtpServer: z.string()
})

export const MailForm: React.FC<MailFormProps> = ({
  isNew,
  onSuccess,
  onDelete
}) => {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)

  const updateEmailSetting = useMutation(updateEmailSettingMutation, {
    onCompleted(data) {
      // todo
    }
  })

  const deleteEmailSetting = useMutation(deleteEmailSettingMutation)

  const handleDelete: React.MouseEventHandler<HTMLButtonElement> = e => {
    e.preventDefault()
    setIsDeleting(true)
    deleteEmailSetting()
      .then(res => {
        if (res?.data?.deleteEmailSetting) {
          onDelete?.()
        } else {
          if (res?.error) {
            toast.error(res?.error?.message)
          }
        }
      })
      .finally(() => {
        setIsDeleting(false)
      })
  }

  return (
    <Form {...form}>
      <div className="flex flex-col items-start gap-4">
        <form
          className="flex flex-col items-start gap-4"
          onSubmit={form.handleSubmit(updateEmailSetting)}
        >
          <FormField
            control={form.control}
            name="smtpServer"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>SMTP Server</FormLabel>
                <FormControl>
                  <Input
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
            name="from"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>From</FormLabel>
                <FormControl>
                  <Input
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
            name="method"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Authentication Method</FormLabel>
                <FormControl>
                  <Select {...field}>
                    <SelectTrigger className="w-80 min-w-max">
                      <SelectValue placeholder="Select a method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="None">None</SelectItem>
                        <SelectItem value="PLAIN">PLAIN</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex gap-8">
            <FormField
              control={form.control}
              name="smtpUsername"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>SMTP Username</FormLabel>
                  <FormControl>
                    <Input
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
              name="smtpPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>SMTP Password</FormLabel>
                  <FormControl>
                    <Input
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
          </div>
          <FormField
            control={form.control}
            name="encryption"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Encryption</FormLabel>
                <FormControl>
                  <Select {...field}>
                    <SelectTrigger className="w-80 min-w-max">
                      <SelectValue placeholder="Select a method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="None">None</SelectItem>
                        <SelectItem value="PLAIN">PLAIN</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="mt-2 flex items-center gap-4">
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
                      {isDeleting && (
                        <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                      )}
                      Yes, delete it
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
            <Button type="submit">{isNew ? 'Create' : 'Update'}</Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
