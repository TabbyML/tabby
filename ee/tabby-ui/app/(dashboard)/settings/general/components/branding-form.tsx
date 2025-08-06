'use client'

import React, { ChangeEvent, useEffect, useState } from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconCloudUpload, IconSpinner } from '@/components/ui/icons'

const updateBrandingSettingMutation = graphql(/* GraphQL */ `
  mutation UpdateBrandingForGeneralSettings($input: BrandingSettingInput!) {
    updateBrandingSetting(input: $input)
  }
`)

const MAX_UPLOAD_SIZE_KB = 500

export const BrandingForm = () => {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [persistedLogo, setPersistedLogo] = useState<string | null>(null)
  const [previewLogo, setPreviewLogo] = useState<string | null>(null)

  useEffect(() => {
    const fetchLogo = async () => {
      setIsLoading(true)
      try {
        const response = await fetch('/branding/logo')
        if (response.ok) {
          const blob = await response.blob()
          const reader = new FileReader()
          reader.onloadend = () => {
            const result = reader.result as string
            // Ensure it's a valid data URL before setting
            if (result && result.startsWith('data:')) {
              setPersistedLogo(result)
            }
          }
          reader.readAsDataURL(blob)
        } else {
          setPersistedLogo(null)
        }
      } catch (error) {
        // Silently fail is ok
      } finally {
        setIsLoading(false)
      }
    }

    fetchLogo()
  }, [])

  const updateBrandingSetting = useMutation(updateBrandingSettingMutation, {
    onError(err) {
      toast.error(err.message)
      setIsSubmitting(false)
    }
  })

  const onPreviewLogo = (e: ChangeEvent<HTMLInputElement>) => {
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
      setPreviewLogo(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleUpdate = async (logo: string | null) => {
    setIsSubmitting(true)
    const response = await updateBrandingSetting({
      input: { brandingLogo: logo }
    })
    if (response?.data?.updateBrandingSetting) {
      toast.success('Successfully updated branding logo!')
      // todo refresh logo
      // await delay(500)
      // window.location.reload()
    } else {
      // Handle case where mutation fails but doesn't throw an error
      setIsSubmitting(false)
    }
  }

  const onRemove = () => {
    if (previewLogo) {
      setPreviewLogo(null)
    } else if (persistedLogo) {
      handleUpdate(null)
    }
  }

  const currentLogo = previewLogo || persistedLogo

  return (
    <div className="grid gap-4">
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
              'opacity-0 hover:opacity-100': currentLogo,
              'hover:bg-background/80': !currentLogo
            }
          )}
        >
          <IconCloudUpload />
          <p className="text-xs text-muted-foreground mt-2">
            {`Accepted file types: .png, .jpg, .webp, .svg. Max file size: ${MAX_UPLOAD_SIZE_KB}KB.`}
          </p>
        </label>
        <input
          id="logo-file"
          type="file"
          accept="image/png, image/jpeg, image/webp, image/svg+xml"
          className="hidden"
          onChange={onPreviewLogo}
        />
        {currentLogo ? (
          <img
            src={currentLogo}
            className="absolute left-0 top-0 z-10 h-full w-full rounded-lg border bg-background object-contain p-2"
            alt="logo"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center rounded-lg border">
            {isLoading ? (
              <IconSpinner className="animate-spin" />
            ) : null}
          </div>
        )}
      </div>

      <div className="flex justify-end">
        <div className="flex items-center gap-x-2">
          {/* {currentLogo && (
            <Button
              variant="ghost"
              type="button"
              onClick={onRemove}
              disabled={isSubmitting}
            >
              <IconTrash className="mr-2 h-4 w-4" />
              Remove
            </Button>
          )} */}
          <Button
            type="submit"
            disabled={!previewLogo || isSubmitting}
            onClick={() => handleUpdate(previewLogo!.split(',')[1])}
          >
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Save Changes
          </Button>
        </div>
      </div>
    </div>
  )
}

