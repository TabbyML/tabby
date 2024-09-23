'use client'

import { ChangeEvent, useState } from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { useMe } from '@/lib/hooks/use-me'
import { useMutation } from '@/lib/tabby/gql'
import { cn, delay } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconCloudUpload, IconSpinner } from '@/components/ui/icons'
import { Separator } from '@/components/ui/separator'
import { mutateAvatar, MyAvatar } from '@/components/user-avatar'

const uploadUserAvatarMutation = graphql(/* GraphQL */ `
  mutation uploadUserAvatarBase64($id: ID!, $avatarBase64: String!) {
    uploadUserAvatarBase64(id: $id, avatarBase64: $avatarBase64)
  }
`)

const MAX_UPLOAD_SIZE_KB = 500

export const Avatar = () => {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [uploadedImgString, setUploadedImgString] = useState('')
  const [{ data }] = useMe()
  const uploadUserAvatar = useMutation(uploadUserAvatarMutation, {
    onError(err) {
      toast.error(err.message)
    }
  })
  if (!data?.me?.email) return null

  const onPreviewAvatar = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files ? e.target.files[0] : null

    if (file) {
      const fileSizeInKB = parseFloat((file.size / 1024).toFixed(2))
      if (fileSizeInKB > MAX_UPLOAD_SIZE_KB) {
        return toast.error(
          `The image you are attempting to upload is too large. Please ensure the file size is under ${MAX_UPLOAD_SIZE_KB}KB and try again.`
        )
      }

      const reader = new FileReader()

      reader.onloadend = () => {
        const imageString = reader.result as string
        setUploadedImgString(imageString)
      }

      reader.readAsDataURL(file)
    }
  }

  const onUploadAvatar = async () => {
    setIsSubmitting(true)

    const response = await uploadUserAvatar({
      avatarBase64: uploadedImgString.split(',')[1],
      id: data.me.id
    })

    if (response?.data?.uploadUserAvatarBase64 === true) {
      await delay(1000)
      mutateAvatar(data.me.id)
      toast.success('Successfully updated your profile picture!')
      await delay(200)
    }

    setUploadedImgString('')
    setIsSubmitting(false)
  }

  return (
    <div className="grid gap-6">
      <div className="relative">
        <label
          htmlFor="avatar-file"
          className="absolute left-0 top-0 z-20 flex h-16 w-16 cursor-pointer items-center justify-center rounded-full bg-background/90 opacity-0 transition-all hover:opacity-100"
        >
          <IconCloudUpload />
        </label>
        <input
          id="avatar-file"
          type="file"
          accept="image/png, image/jpeg"
          className="hidden"
          onChange={onPreviewAvatar}
        />
        {uploadedImgString && (
          <img
            src={uploadedImgString}
            className="absolute left-0 top-0 z-10 h-16 w-16 rounded-full border object-cover"
            alt="avatar to be uploaded"
          />
        )}
        <MyAvatar
          className={cn('relative h-16 w-16 border', {
            'opacity-0': uploadedImgString
          })}
        />
      </div>

      <Separator />

      <div className="flex items-center justify-between">
        <Button
          type="submit"
          disabled={!uploadedImgString || isSubmitting}
          onClick={onUploadAvatar}
          className="mr-5 w-40"
        >
          {isSubmitting && (
            <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
          )}
          Save Changes
        </Button>

        <div className="flex flex-1 justify-end">
          <p className=" text-xs text-muted-foreground lg:text-sm">
            {`Square image recommended. Accepted file types: .png, .jpg. Max file size: ${MAX_UPLOAD_SIZE_KB}KB.`}
          </p>
        </div>
      </div>
    </div>
  )
}
