'use client'

import { ChangeEvent  } from 'react'

import { cn } from '@/lib/utils'
import { useMe } from '@/lib/hooks/use-me'
import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { toast } from 'sonner'

import { buttonVariants } from "@/components/ui/button"
import { UserAvatar, mutateAvatar } from '@/components/user-avatar'

const uploadUserAvatarMutation = graphql(/* GraphQL */ `
  mutation uploadUserAvatarBase64($id: ID!, $avatarBase64: String!) {
    uploadUserAvatarBase64(id: $id, avatarBase64: $avatarBase64)
  }
`)

export const Avatar = () => {
  const [{ data }] = useMe()
  const uploadUserAvatar = useMutation(uploadUserAvatarMutation)
  if (!data?.me?.email) return null

  const onUploadAvatar = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files ? e.target.files[0] : null;
    if (file) {
      const reader = new FileReader();

      reader.onloadend = async () => {
        try {
          const imageString = reader.result as string;
          const mimeHeaderMatcher = new RegExp("^data:image/.+;base64,")
          const avatarBase64 = imageString.replace(mimeHeaderMatcher, "")

          const response = await uploadUserAvatar({
            avatarBase64,
            id: data.me.id
          })
          if (response?.error) throw response.error
          if (response?.data?.uploadUserAvatarBase64 === false) throw new Error('Upload failed')
          mutateAvatar(data.me.id)
          toast.success('Avatar uploaded successfully.')
        } catch (err: any) {
          toast.error(err.message || 'Upload failed')
        }
      };

      reader.readAsDataURL(file);
    }
  }

  return (
    <div className="flex items-center">
      <UserAvatar className="h-16 w-16 border" />

      <div className="ml-3">
        <label htmlFor="avatar-file" className={cn("relative cursor-pointer", buttonVariants({ variant: 'outline' }))}>Upload new picture</label>
        <input id="avatar-file" type="file" accept='image/*' className="hidden" onChange={onUploadAvatar} />
        <p className="mt-1.5 text-xs text-muted-foreground">Recommended: Square JPG, PNG, at least 1,000 pixels per side.</p>
      </div>
    </div>
  )
}
