'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useSignIn } from '@/lib/tabby/auth'
import { useGraphQLForm } from '@/lib/tabby/gql'
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

const createInvitation = graphql(/* GraphQL */ `
  mutation CreateInvitation($email: String!) {
    createInvitation(email: $email)
  }
`)

const formSchema = z.object({
    email: z.string().email('Invalid email address'),
})

export default function CreateInvitationForm({ onCreated }: { onCreated: () => void }) {
    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
    })

    const { isSubmitting } = form.formState
    const { onSubmit } = useGraphQLForm(createInvitation, {
        onSuccess: () => {
            form.reset({ email: "" })
            onCreated()
        },
        onError: (path, message) => form.setError(path as any, { message })
    })

    return (<Form {...form}>
        <div className='flex flex-col items-end gap-2'>
            <form className="flex gap-4 items-center" onSubmit={form.handleSubmit(onSubmit)}>
                <FormField
                    control={form.control}
                    name="email"
                    render={({ field }) => (
                        <FormItem>
                            <FormControl>
                                <Input
                                    placeholder="Email"
                                    type="email"
                                    autoCapitalize="none"
                                    autoComplete="email"
                                    autoCorrect="off"
                                    {...field}
                                />
                            </FormControl>
                        </FormItem>
                    )}
                />
                <Button type="submit" disabled={isSubmitting}>
                    Invite
                </Button>
            </form>
            <FormMessage className="text-center" />
        </div>
    </Form>
    )
}
