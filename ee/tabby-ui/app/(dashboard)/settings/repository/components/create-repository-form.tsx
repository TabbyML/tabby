'use client'

import * as React from 'react'
import Link from 'next/link'
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
  FormLabel,
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
      form.reset({ name: undefined, gitUrl: undefined })
      onCreated()
    },
    form
  })

  return (
    <Form {...form}>
      <div className="flex flex-col items-start gap-2">
        <form
          className="flex flex-col w-full gap-4"
          onSubmit={form.handleSubmit(createRepository)}
        >
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Name</FormLabel>
                <FormControl>
                  <Input
                    autoCapitalize="none"
                    autoCorrect="off"
                    autoComplete="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="gitUrl"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Git URL</FormLabel>
                <FormControl>
                  <Input autoCapitalize="none" autoCorrect="off" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormMessage className="text-center" />
          <div className="flex justify-end gap-4">
            <Link href="/settings/repository">
              <Button type="button" variant="ghost" disabled={isSubmitting}>
                Cancel
              </Button>
            </Link>
            <Button type="submit" disabled={isSubmitting}>
              Create
            </Button>
          </div>
        </form>
      </div>
    </Form>
  )
}
