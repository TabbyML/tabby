'use client'

import { useRef, useState, useEffect } from 'react'
import type { ChatMessage, Context, FetcherOptions } from 'tabby-chat-panel'
import { useServer } from 'tabby-chat-panel/react'
import { useSearchParams } from 'next/navigation'

import { nanoid } from '@/lib/utils'
import { Chat, ChatRef } from '@/components/chat/chat'
import Color from 'color'

import './page.css'

const convertToHSLColor = (style: string) => {
  return Color(style)
    .hsl()
    .toString()
    .replace(/hsla?\(/, '')
    .replace(')', '')
    .split(',')
    .slice(0, 3)
    .map((item, idx) => {
      if (idx === 0) return parseFloat(item).toFixed(1)
      return item
    })
    .join('')
}

export default function ChatPage() {
  const [isInit, setIsInit] = useState(false)
  const [fetcherOptions, setFetcherOptions] = useState<FetcherOptions | null>(
    null
  )
  const [activeChatId, setActiveChatId] = useState('')
  const [pendingMessages, setPendingMessages] = useState<ChatMessage[]>([])

  const chatRef = useRef<ChatRef>(null)

  const searchParams = useSearchParams()
  const from = searchParams.get('from') || undefined
  const isFromVSCode = from === 'vscode'
  const maxWidth = isFromVSCode ? '5xl' : undefined

  useEffect(() => {
    const onMessage = ({
      data
    }: {
      data: {
        style?: string
        themeClass?: string
      }
    }) => {
      // Sync with VSCode CSS variable
      if (data.style) {
        const styleWithHslValue = data.style
          .split(';')
          .filter((style: string) => style)
          .map((style: string) => {
            const [key, value] = style.split(':')
            const styleValue = value.trim()
            const isColorValue =
              styleValue.startsWith('#') || styleValue.startsWith('rgb')
            if (!isColorValue) return `${key}: ${value}`
            const hslValue = convertToHSLColor(styleValue)
            return `${key}: ${hslValue}`
          })
          .join(';')
        document.documentElement.style.cssText = styleWithHslValue
      }

      // Sync with edit theme
      if (data.themeClass) {
        document.documentElement.className = data.themeClass
      }
    }

    window.addEventListener('message', onMessage)
    return () => {
      window.removeEventListener('message', onMessage)
    }
  }, [])

  // VSCode bug: not support shortcuts like copy/paste
  // @see - https://github.com/microsoft/vscode/issues/129178
  useEffect(() => {
    if (!isFromVSCode) return

    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.code === 'KeyC') {
        document.execCommand('copy')
      } else if ((event.ctrlKey || event.metaKey) && event.code === 'KeyX') {
        document.execCommand('cut')
      } else if ((event.ctrlKey || event.metaKey) && event.code === 'KeyV') {
        document.execCommand('paste')
      } else if ((event.ctrlKey || event.metaKey) && event.code === 'KeyA') {
        document.execCommand('selectAll')
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  const sendMessage = (message: ChatMessage) => {
    if (chatRef.current) {
      chatRef.current.sendUserChat(message)
    } else {
      const newPendingMessages = [...pendingMessages]
      newPendingMessages.push(message)
      setPendingMessages(newPendingMessages)
    }
  }

  const server = useServer({
    init: request => {
      if (chatRef.current) return
      setActiveChatId(nanoid())
      setIsInit(true)
      setFetcherOptions(request.fetcherOptions)
    },
    sendMessage: (message: ChatMessage) => {
      return sendMessage(message)
    }
  })

  const onChatLoaded = () => {
    pendingMessages.forEach(sendMessage)
    setPendingMessages([])
  }

  const onNavigateToContext = (context: Context) => {
    server?.navigate(context)
  }

  // if (!isInit || !fetcherOptions) return <></>
  // const headers = {
  //   Authorization: `Bearer ${fetcherOptions.authorization}`
  // }
  return (
    <div className="h-screen overflow-auto relative">
      <Chat
        chatId={activeChatId}
        key={activeChatId}
        ref={chatRef}
        // headers={headers}
        onThreadUpdates={() => {}}
        onNavigateToContext={onNavigateToContext}
        onLoaded={onChatLoaded}
        maxWidth={maxWidth}
        initialMessages={[
          {
            user: {
              message: 'How to convert a list of string to numbers in python',
              id: 'bGYu135'
            },
            assistant: {
              id: 'iGtRdsG',
              message:
                "To convert a list of string to numbers in Python, we can use the `map()` function along with the `float()` function. Here's an example:\n\n```python\nmy_list = ['1', '2', '3', '4', '5']\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, 5.0]\n```\n\nIn this example, we first define a list of strings `my_list`. Then, we use the `map()` function to convert each string in the list to a float. Finally, we convert the resulting float list back to a list using the `list()` function and print it.\n\nNote that this method will only work for lists of strings that can be converted to floats. If you have a list of numbers, you can simply convert it to a list of floats using the `list()` function. For example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, 5.0]\n```\n\nIn this example, we first define a list of numbers `my_list`. Then, we use the `map()` function to convert each number in the list to a float. Finally, we convert the resulting float list back to a list using the `list()` function and print it.-string-to-numbers-in-python-60454270c6b8\n\nIn this article, we will learn how to convert a list of string to numbers in Python using the `map()` function and the `float()` function. We will also see how to convert a list of numbers to a list of floats using the `list()` function.\n\n## Convert a list of string to numbers in Python\n\nTo convert a list of string to numbers in Python, we can use the `map()` function along with the `float()` function. Here's an example:\n\n```python\nmy_list = ['1', '2', '3', '4', '5']\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, 5.0]\n```\n\nIn this example, we first define a list of strings `my_list`. Then, we use the `map()` function to convert each string in the list to a float. Finally, we convert the resulting float list back to a list using the `list()` function and print it.\n\nNote that this method will only work for lists of strings that can be converted to floats. If you have a list of numbers, you can simply convert it to a list of floats using the `list()` function. For example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, 5.0]\n```\n\nIn this example, we first define a list of numbers `my_list`. Then, we use the `map()` function to convert each number in the list to a float. Finally, we convert the resulting float list back to a list using the `list()` function and print it.\n\n## Convert a list of numbers to a list of floats in Python\n\nTo convert a list of numbers to a list of floats in Python, we can simply convert the list to a list of floats using the `list()` function. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, 5.0]\n```\n\nIn this example, we first define a list of numbers `my_list`. Then, we convert the list to a float list using the `list()` function and print it. Note that this method will only work for lists of numbers that can be converted to floats. If you have a list of strings, you can simply convert it to a list of floats using the `list()` function. For example:\n\n```python\nmy_list = ['1', '2', '3', '4', '5']\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, 5.0]\n```\n\nIn this example, we first define a list of strings `my_list`. Then, we convert the list to a float list using the `list()` function and print it. Note that this method will only work for lists of strings that can be converted to floats. If you have a list of numbers, you can simply convert it to a list of floats using the `list()` function. For example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, 5.0]\n```\n\nIn this example, we first define a list of numbers `my_list`. Then, we convert the list to a float list using the `list()` function and print it. Note that this method will only work for lists of numbers that can be converted to floats. If you have a list of strings, you can simply convert it to a list of floats using the `list()` function. For example:\n\n```python\nmy_list = ['1', '2', '3', '4', '5']\n\n# convert the list to a float list\nfloat_list = list(map(float, my_list))\n\nprint(float_list)\n```\n\nOutput:\n\n```\n[1.0, 2.0, 3.0, 4.0, "
            }
          },
          {
            user: {
              message:
                'what is the difference between number of js and number of python',
              id: '2vzalzb'
            },
            assistant: {
              id: 'yOnNu5w',
              message:
                "In JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `typeof` operator to determine the type of a variable. In Python, we can use the `type()` function to determine the type of a variable. Here's an example:\n\n```python\nmy_list = [1, 2, 3, 4, 5]\n\n# determine the type of the list\nprint(type(my_list))\n```\n\nOutput:\n\n```\n<class 'list'>\n```\n\nIn this example, we first define a list `my_list`. Then, we use the `type()` function to determine the type of the list. The output shows that the list is a list.\n\nIn JavaScript, we can use the `"
            }
          }
        ]}
      />
    </div>
  )
}
