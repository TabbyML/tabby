'use client'

import { ChangeEvent, useState } from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { useMe } from '@/lib/hooks/use-me'
import { useMutation } from '@/lib/tabby/gql'
import { delay } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconCloudUpload, IconSpinner } from '@/components/ui/icons'
import { Separator } from '@/components/ui/separator'
import { mutateAvatar, UserAvatar } from '@/components/user-avatar'

const uploadUserAvatarMutation = graphql(/* GraphQL */ `
  mutation uploadUserAvatarBase64($id: ID!, $avatarBase64: String!) {
    uploadUserAvatarBase64(id: $id, avatarBase64: $avatarBase64)
  }
`)

export const Avatar = () => {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [uploadedImgString, setUploadedImgString] = useState('')
  const [{ data }] = useMe()
  const uploadUserAvatar = useMutation(uploadUserAvatarMutation)
  if (!data?.me?.email) return null

  const onPreviewAvatar = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files ? e.target.files[0] : null
    if (file) {
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
    try {
      const mimeHeaderMatcher = new RegExp('^data:image/.+;base64,')
      const avatarBase64 = uploadedImgString.replace(mimeHeaderMatcher, '')

      const response = await uploadUserAvatar({
        avatarBase64,
        id: data.me.id
      })
      if (response?.error) throw response.error
      if (response?.data?.uploadUserAvatarBase64 === false)
        throw new Error('Upload failed')
      await delay(1000)
      mutateAvatar(data.me.id)
      toast.success('Avatar uploaded successfully.')
      await delay(200)
      setUploadedImgString('')
    } catch (err: any) {
      toast.error(err.message || 'Upload failed')
    }
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
          accept="image/*"
          className="hidden"
          onChange={onPreviewAvatar}
        />
        {uploadedImgString && (
          <img
            src={uploadedImgString}
            className="absolute left-0 top-0 z-10 h-16 w-16 rounded-full border object-cover"
            alt="Upload image preview"
          />
        )}
        <UserAvatar className="relative h-16 w-16 border" />
      </div>

      <Separator />

      <div className="flex justify-between">
        <Button
          type="submit"
          disabled={!uploadedImgString || isSubmitting}
          onClick={onUploadAvatar}
          className="w-40"
        >
          {isSubmitting && (
            <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
          )}
          Update avatar
        </Button>

        <p className="mt-1.5 text-sm text-muted-foreground">
          Square image recommended. Accept file types: .png, .jpg.
        </p>
      </div>
    </div>
  )
}
