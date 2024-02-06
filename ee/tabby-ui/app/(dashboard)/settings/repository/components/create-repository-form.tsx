'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'

const createRepositoryMutation = graphql(/* GraphQL */ `
  mutation createRepository($name: String!, $gitUrl: String!) {
    createRepository(name: $name, gitUrl: $gitUrl)
  }
`)

const formSchema = z.object({
  name: z.string(),
  gitUrl: z.string()
})

export default function CreateRepositoryForm({
  onCreated
}: {
  onCreated: () => void
}) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState
  const createRepository = useMutation(createRepositoryMutation, {
    onCompleted() {
      form.reset({ name: '', gitUrl: '' })
      onCreated()
    },
    form
  })

  return (
    <Form {...form}>
      <div className="flex flex-col items-start gap-2">
        <form
          className="flex w-full items-center gap-2"
          onSubmit={form.handleSubmit(createRepository)}
        >
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input
                    autoCapitalize="none"
                    autoCorrect="off"
                    autoComplete="off"
                    {...field}
                  />
                </FormControl>
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="gitUrl"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Input autoCapitalize="none" autoCorrect="off" {...field} />
                </FormControl>
              </FormItem>
            )}
          />
          <Button type="submit" disabled={isSubmitting}>
            Add Git Repository
          </Button>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
