'use client'

import { Button } from "@/components/ui/button"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { graphql } from "@/lib/gql/generates"
import { useForm } from "react-hook-form"


const updateEmailSettingMutation = graphql(/* GraphQL */ `
  mutation updateEmailSetting($smtpUsername: String!, $smtpPassword: String, $smtpServer: String!) {
    updateEmailSetting(smtpUsername: $smtpUsername, smtpPassword: $smtpPassword, smtpServer: $smtpServer)
  }
`)

const deleteEmailSettingMutation = graphql(/* GraphQL */ `
  mutation updateEmailSetting {
    updateEmailSetting
  }
`)

export const MailForm = () => {


  const form = useForm()

  return (
    <Form {...form}>
      <div className="flex flex-col items-start gap-4">
        <form
          className="flex flex-col items-start gap-4"
        // onSubmit={form.handleSubmit(createInvitation)}
        >
          <FormField
            control={form.control}
            name="from"
            render={({ field }) => (
              <FormItem>
                {/* todo required */}
                <FormLabel>From</FormLabel>
                <FormControl>
                  <Input
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    className="w-[320px]"
                    {...field}
                  />
                </FormControl>
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
                    <SelectTrigger className="w-[320px]">
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
              </FormItem>
            )}
          />
          <div className="grid grid-cols-2 gap-8">
            <FormField
              control={form.control}
              name="smtpUsername"
              render={({ field }) => (
                <FormItem>
                  {/* todo required */}
                  <FormLabel>SMTP Username</FormLabel>
                  <FormControl>
                    <Input
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      className="w-[320px]"
                      {...field}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="smtpPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>SMTP Password</FormLabel>
                  <FormControl>
                    <Input
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      {...field}
                    />
                  </FormControl>
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
                    <SelectTrigger className="w-[320px]">
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
              </FormItem>
            )}
          />
          <Button type="submit">
            Update
          </Button>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
