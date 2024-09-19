'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { NotionDocumentType } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
  
} from '@/components/ui/form'
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'

const createNotionDocumentMutation = graphql(/* GraphQL */ `
  mutation CreateNotionDocument($input: CreateNotionDocumentInput!) {
    createNotionDocument(input: $input)
  }
`)

// const intergrationTypeSchema = z.enum([
//   NotionDocumentType.Database,
// ])

const formSchema = z.object({
  accessToken: z.string().trim(),
  integrationId: z.string().trim(),
  integrationType:  z.nativeEnum(NotionDocumentType),
  name: z.string().trim(),
})

type FormValues = z.infer<typeof formSchema>

export default function CreateNotionDocument() {
  const router = useRouter()
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const onCreated = () => {
    router.push('./')
  }

  const { isSubmitting } = form.formState
  const createNotionDocument = useMutation(createNotionDocumentMutation, {
    onCompleted() {
      onCreated()
    },
    form
  })

  const onSubmit = (values: FormValues) => {
    return createNotionDocument({
      input: {
        accessToken: values.accessToken,
        integrationId: values.integrationId,
        integrationType: values.integrationType,
        name: values.name
      }
    })
  }

  return (
    <>
      <Form {...form}>
        <div className="grid gap-2">
          <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Name</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. tabby notion"
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="accessToken"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Access Token</FormLabel>
                  <FormControl>
                    <Input
                       placeholder='e.g. secret_******'
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              
              )}
            />
            <FormField
              control={form.control}
              name="integrationId"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Integration ID</FormLabel>
                  <FormControl>
                    <Input
                     
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="integrationType"
              render={({ field }) => (

                <FormItem>
                  <FormLabel required>Integration Type</FormLabel>
                  <Select
                    value={NotionDocumentType.Database}
                    //disabled={true}
                    defaultValue={NotionDocumentType.Database}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Integration Type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NotionDocumentType.Database}>DATABASE</SelectItem>
                    </SelectContent>
                  </Select>
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
    </>
  )
}
