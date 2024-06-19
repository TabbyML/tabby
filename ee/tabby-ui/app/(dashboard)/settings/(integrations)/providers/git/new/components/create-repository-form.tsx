'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'

const createRepositoryMutation = graphql(/* GraphQL */ `
  mutation createGitRepository($name: String!, $gitUrl: String!) {
    createGitRepository(name: $name, gitUrl: $gitUrl)
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

  const router = useRouter()
  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form
          className="grid gap-6"
          onSubmit={form.handleSubmit(createRepository)}
        >
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Name</FormLabel>
                <FormControl>
                  <Input
                    placeholder="e.g. tabby"
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
                <FormLabel required>Git URL</FormLabel>
                <FormDescription>Remote or local Git URL</FormDescription>
                <FormControl>
                  <Input
                    placeholder="e.g. https://github.com/TabbyML/tabby"
                    autoCapitalize="none"
                    autoCorrect="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex justify-end gap-4">
            <Button
              type="button"
              variant="ghost"
              disabled={isSubmitting}
              onClick={() => router.back()}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              Create
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
