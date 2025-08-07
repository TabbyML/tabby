'use client'

import React, { ChangeEvent } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import { useQuery } from 'urql'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { brandingSettingQuery } from '@/lib/tabby/query'
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
import {
  IconCloudUpload,
  IconClose,
  IconSpinner
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
// import { mutateBranding } from '@/components/branding-logo'
import LoadingWrapper from '@/components/loading-wrapper'
import { FormSkeleton } from '@/components/skeleton'

const updateBrandingSettingMutation = graphql(/* GraphQL */ `
  mutation GeneralBrandingMutation($input: BrandingSettingInput!) {
    updateBrandingSetting(input: $input)
  }
`)

const formSchema = z.object({
  brandingName: z.string().optional(),
  brandingLogo: z.string().optional(),
  brandingIcon: z.string().optional()
})

const MAX_UPLOAD_SIZE_KB = 500

interface BrandingFormProps {
  defaultValues?: Partial<BrandingFormValues>
  onSuccess?: () => void
}

type BrandingFormValues = z.infer<typeof formSchema>

const BrandingForm: React.FC<BrandingFormProps> = ({
  defaultValues,
  onSuccess
}) => {
  const form = useForm<BrandingFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues
  })

  const { brandingLogo, brandingIcon } = form.watch()

  const updateBrandingSetting = useMutation(updateBrandingSettingMutation, {
    form,
    onCompleted(values) {
      if (values?.updateBrandingSetting) {
        toast.success('Branding settings saved!')
        onSuccess?.()
        form.reset(form.getValues())
      }
    }
  })

  const onFileChange = (
    e: ChangeEvent<HTMLInputElement>,
    field: 'brandingLogo' | 'brandingIcon'
  ) => {
    const file = e.target.files?.[0]
    if (!file) return

    const fileSizeInKB = parseFloat((file.size / 1024).toFixed(2))
    if (fileSizeInKB > MAX_UPLOAD_SIZE_KB) {
      toast.error(
        `The image you are attempting to upload is too large. Please ensure the file size is under ${MAX_UPLOAD_SIZE_KB}KB and try again.`
      )
      return
    }

    const reader = new FileReader()
    reader.onloadend = () => {
      form.setValue(field, reader.result as string, { shouldDirty: true })
    }
    reader.readAsDataURL(file)
  }

  const removeImage = (field: 'brandingLogo' | 'brandingIcon') => {
    form.setValue(field, "", { shouldDirty: true })
  }

  const onSubmit = async (values: BrandingFormValues) => {
    await updateBrandingSetting({
      input: {
        ...values,
        brandingLogo:
          values.brandingLogo === ''
            ? null
            : values.brandingLogo?.startsWith('data:')
            ? values.brandingLogo
            : undefined,
        brandingIcon:
          values.brandingIcon === ''
            ? null
            : values.brandingIcon?.startsWith('data:')
            ? values.brandingIcon
            : undefined
      }
    })
  }

  return (
    <Form {...form}>
      <form className="grid gap-4" onSubmit={form.handleSubmit(onSubmit)}>
        <FormField
          control={form.control}
          name="brandingName"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Brand Name</FormLabel>
              <FormControl>
                <Input {...field} className="w-[26rem]" />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <div className="flex flex-col gap-y-2">
          <div>Logo</div>
          <p className="text-sm text-muted-foreground">
            The suggested logo size should be 5:2 aspect ratio, e.g 100 x 40.
          </p>
        </div>
        <div className="relative h-36 w-[26rem]">
          <label
            htmlFor="logo-file"
            className={cn(
              'absolute left-0 top-0 z-20 flex h-full w-full cursor-pointer flex-col items-center justify-center gap-y-2 rounded-lg border-2 border-dashed bg-background/90 transition-opacity',
              {
                'opacity-0 hover:opacity-100': brandingLogo,
                'hover:bg-background/80': !brandingLogo
              }
            )}
          >
            <IconCloudUpload />
            <p className="mt-2 text-xs text-muted-foreground">
              {`Accepted file types: .png, .jpg, .webp, .svg. Max file size: ${MAX_UPLOAD_SIZE_KB}KB.`}
            </p>
          </label>
          <input
            id="logo-file"
            type="file"
            accept="image/png, image/jpeg, image/webp, image/svg+xml"
            className="hidden"
            onChange={e => onFileChange(e, 'brandingLogo')}
          />
          {brandingLogo ? (
            <div className="relative h-full w-full">
              <img
                src={brandingLogo}
                className="absolute left-0 top-0 z-10 h-full w-full rounded-lg border bg-background object-contain p-2"
                alt="logo"
              />
              <Button
                type="button"
                onClick={() => removeImage('brandingLogo')}
                variant='hover-destructive'
                className="absolute -right-2 -top-2 z-20 cursor-pointer rounded-full p-0.5 h-auto border bg-background"
              >
                <IconClose className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="flex h-full w-full items-center justify-center rounded-lg border" />
          )}
        </div>

        <div className="flex flex-col gap-y-2">
          <div>Icon</div>
          <p className="text-sm text-muted-foreground">
            The suggested icon size should be square, e.g 40 x 40.
          </p>
        </div>
        <div className="relative h-36 w-36">
          <label
            htmlFor="icon-file"
            className={cn(
              'absolute left-0 top-0 z-20 flex h-full w-full cursor-pointer flex-col items-center justify-center gap-y-2 rounded-lg border-2 border-dashed bg-background/90 transition-opacity',
              {
                'opacity-0 hover:opacity-100': brandingIcon,
                'hover:bg-background/80': !brandingIcon
              }
            )}
          >
            <IconCloudUpload />
            <p className="mt-2 text-xs text-muted-foreground">
              {`Max file size: ${MAX_UPLOAD_SIZE_KB}KB.`}
            </p>
          </label>
          <input
            id="icon-file"
            type="file"
            accept="image/png, image/jpeg, image/webp, image/svg+xml"
            className="hidden"
            onChange={e => onFileChange(e, 'brandingIcon')}
          />
          {brandingIcon ? (
            <div className="relative h-full w-full">
              <img
                src={brandingIcon}
                className="absolute left-0 top-0 z-10 h-full w-full rounded-lg border bg-background object-contain p-2"
                alt="icon"
              />
              <Button
                type="button"
                onClick={() => removeImage('brandingIcon')}
                variant='hover-destructive'
                className="absolute -right-2 -top-2 z-20 cursor-pointer rounded-full p-0.5 h-auto border bg-background"
              >
                <IconClose className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="flex h-full w-full items-center justify-center rounded-lg border" />
          )}
        </div>

        <div className="flex justify-end">
          <div className="flex items-center gap-x-2">
            <Button
              type="submit"
              disabled={!form.formState.isDirty || form.formState.isSubmitting}
            >
              {form.formState.isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Update
            </Button>
          </div>
        </div>
      </form>
    </Form>
  )
}

export const GeneralBrandingForm = () => {
  const [{ data, stale }, reexecuteQuery] = useQuery({
    query: brandingSettingQuery
  })

  const onSuccess = () => {
    reexecuteQuery({ requestPolicy: 'network-only' })
  }

  return (
    <div className="min-h-[160px]">
      <LoadingWrapper loading={!data || stale} fallback={<FormSkeleton />}>
        <BrandingForm
          defaultValues={{
            brandingName: data?.brandingSetting?.brandingName ?? undefined,
            brandingLogo: data?.brandingSetting?.brandingLogo ?? undefined,
            brandingIcon: data?.brandingSetting?.brandingIcon ?? undefined
          }}
          onSuccess={onSuccess}
        />
      </LoadingWrapper>
    </div>
  )
}
