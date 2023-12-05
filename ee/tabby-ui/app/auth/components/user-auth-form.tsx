"use client"

import * as React from "react"

import { zodResolver } from "@hookform/resolvers/zod"
import { FieldError, FieldErrors, useForm, useFormContext, useFormState } from "react-hook-form"
import * as z from "zod"

import { cn } from "@/lib/utils"
import { IconSpinner } from "@/components/ui/icons"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { graphql } from "@/lib/gql/generates"
import { useGraphQLForm } from "@/lib/tabby-gql-client"

export const registerUser = graphql(/* GraphQL */ `
  mutation register($email: String!, $password1: String!, $password2: String!, $invitationCode: String) {
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

const formSchema = z.object({
  email: z.string().email("Invalid email address"),
  password1: z.string(),
  password2: z.string(),
  invitationCode: z.string().optional(),
});

interface UserAuthFormProps extends React.HTMLAttributes<HTMLDivElement> {
  invitationCode?: string
}

export function UserAuthForm({ className, invitationCode, ...props }: UserAuthFormProps) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: "",
      password1: "",
      password2: "",
      invitationCode,
    }
  })

  const { isSubmitting } = form.formState;
  const { onSubmit } = useGraphQLForm(registerUser, {
    onError: (path, message) => form.setError(path as any, { message })
  });

  return (
    <div className={cn("grid gap-6", className)} {...props}>
      <Form {...form}>
        <form className="grid gap-2" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField control={form.control} name="email" render={({ field }) => (
            <FormItem>
              <FormLabel>Email</FormLabel>
              <FormControl>
                <Input
                  placeholder=""
                  type="email"
                  autoCapitalize="none"
                  autoComplete="email"
                  autoCorrect="off"
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )} />
          <FormField control={form.control} name="password1" render={({ field }) => (
            <FormItem>
              <FormLabel>Password</FormLabel>
              <FormControl>
                <Input type="password" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )} />
          <FormField control={form.control} name="password2" render={({ field }) => (
            <FormItem>
              <FormLabel>Confirm Password</FormLabel>
              <FormControl>
                <Input type="password" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )} />
          <FormField control={form.control} name="invitationCode" render={({ field }) => (
            <FormItem>
              <FormControl>
                <Input type="hidden" {...field} />
              </FormControl>
            </FormItem>
          )} />
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Register
          </Button>
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}