import './styles.css'

import Document from '@tiptap/extension-document'
import Mention from '@tiptap/extension-mention'
import Paragraph from '@tiptap/extension-paragraph'
import Text from '@tiptap/extension-text'
import { EditorContent, Extension, useEditor } from '@tiptap/react'
import React, { useState } from 'react'
import Placeholder from "@tiptap/extension-placeholder";
import StarterKit from '@tiptap/starter-kit'
import Highlight from '@tiptap/extension-highlight'
import Typography from '@tiptap/extension-typography'

import suggestion from './suggestion'

const DisableEnter = Extension.create({
  addKeyboardShortcuts() {
    return {
      Enter: ({ editor }) => {
        console.log('submit')
        return true
      }
    };
  },
})

export const PromptEditor = () => {
  const [count, setCount] = useState(1)
  const editor = useEditor({
    extensions: [
      StarterKit,
      Highlight,
      Typography,
      DisableEnter,
      Mention.configure({
        HTMLAttributes: {
          class: 'mention',
        },
        suggestion,
      }),
      Placeholder.configure({
        placeholder: "Ask anything...",
      }),
    ],
    editorProps: {
      attributes: {
        class: "prose dark:prose-invert prose-p:my-0 focus:outline-none max-w-none max-h-38 pt-5",
      },
      handleDOMEvents: {
        keydown: (_, event) => {
          if (event.key === 'Enter' && event.shiftKey) {
            if (editor) {
              editor.commands.setHardBreak()
            }
            event.preventDefault()
            return true
          }
        },
      }
    },
    content: `
      who are you?
      
      And tell me
    `,
  })

  if (!editor) {
    return null
  }

  const getMentionsWithIndices = () => {
    const json = editor.getJSON();
    const mentions = [];
    let textLength = 0;

    const traverse = (node) => {
      if (node.type === 'text') {
        textLength += node.text.length;
      } else if (node.type === 'mention') {
        mentions.push({
          id: node.attrs.id,
          start: textLength,
        });
        textLength += (node.attrs.label || node.attrs.id).length;
      } else if (node.type === 'hardBreak') {
        textLength += 1; // Assuming hardBreak is represented as a single character
      }

      if (node.content) {
        node.content.forEach(traverse);
      }
    };

    traverse(json);
    return mentions;
  };

  return (
    <div className='text-area-autosize pr-1 max-h-36 overflow-y-auto'
    >
      <EditorContent
        editor={editor}
        onBlur={e => {
          setCount(c => c + 1)
          console.log(editor.getText())
          console.log(editor.getJSON())
          console.log(editor.getHTML())
          console.log(getMentionsWithIndices())
        }}
      />
    </div>
  )
}