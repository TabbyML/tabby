// import React, { useEffect } from 'react'
// import Editor, { EditorProps, useMonaco } from '@monaco-editor/react'
// import { editor } from 'monaco-editor'
// import { useTheme } from 'next-themes'
// import { createRoot } from 'react-dom/client'

// import { Button } from './ui/button'
// import { Popover, PopoverContent, PopoverTrigger } from './ui/popover'

// // import { createClient } from '@opencodegraph/client'
// // import { createExtension, makeRange } from '@opencodegraph/monaco-editor-extension'

// // // Set up a client.
// // const client = createClient({
// //   configuration: () =>
// //     Promise.resolve({
// //       enable: true,
// //       providers: {
// //         'https://opencodegraph.org/npm/@opencodegraph/provider-hello-world': true,
// //       },
// //     }),
// //   makeRange,
// //   logger: console.error,
// // })

// interface Props extends EditorProps {}

// export const MonacoEditor: React.FC<Props> = ({ language, ...props }) => {
//   const monaco = useMonaco()
//   const { theme } = useTheme()
//   const editorRef = React.useRef<editor.IStandaloneCodeEditor>(null)
//   const finalTheme = theme === 'dark' ? 'vs-dark' : theme

//   useEffect(() => {
//     if (!monaco) return
//     monaco.languages.typescript.typescriptDefaults.setDiagnosticsOptions({
//       noSemanticValidation: true,
//       noSyntaxValidation: true // This line disables errors in jsx tags like <div>, etc.
//     })
//     monaco.languages.typescript.javascriptDefaults.setCompilerOptions({
//       // jsx: "react",
//       tsx: 'react'
//     })
//   }, [monaco])

//   useEffect(() => {
//     if (!monaco || !editorRef.current) return

//     const CustomCom = () => {
//       return (
//         <Popover>
//           <PopoverTrigger>
//             <Button>tstttt</Button>
//           </PopoverTrigger>
//           <PopoverContent>
//             <div>content</div>
//           </PopoverContent>
//         </Popover>
//       )
//     }
//     const widget = {
//       getId: function () {
//         return 'my.content.widget'
//       },
//       getDomNode: function () {
//         const domNode = document.createElement('div')
//         const root = createRoot(domNode)
//         root.render(<CustomCom />)
//         return domNode
//       },
//       getPosition: function () {
//         return {
//           position: {
//             lineNumber: 7,
//             column: 8
//           },
//           preference: [
//             monaco.editor.ContentWidgetPositionPreference.ABOVE,
//             monaco.editor.ContentWidgetPositionPreference.BELOW
//           ]
//         }
//       }
//     }

//     monaco.languages.registerCodeLensProvider(['*'], {
//       provideCodeLenses: function (model, token) {
//         return {
//           lenses: [
//             {
//               range: new monaco.Range(1, 2, 1, 1), // 将组件放置在第一行
//               id: 'unique-id-for-your-component',
//               command: {
//                 id: 'unique-command-id',
//                 title: 'Your Component Title',
//                 tooltip: '🚀Tooltip for your component \n hahaha',
//                 arguments: [
//                   /* add any arguments if needed */
//                 ]
//               }
//             }
//           ],
//           dispose: () => {}
//         }
//       },
//       resolveCodeLens: function (model, codeLens, token) {
//         // 如果需要在用户点击组件时执行进一步的操作，可以在这里进行处理
//         return codeLens
//       }
//     })

//     setTimeout(() => {
//       editorRef.current.addContentWidget(widget)
//     }, 10000)
//   }, [monaco, editorRef.current])

//   function handleEditorDidMount(editor, monaco) {
//     // here is the editor instance
//     // you can store it in `useRef` for further usage
//     editorRef.current = editor
//   }

//   return (
//     <Editor
//       onMount={handleEditorDidMount}
//       {...props}
//       theme={finalTheme}
//       language={language}
//     />
//   )
// }

// // todo get lang from lang and file extname
// export function getLanguage(lang: string = ''): string {
//   switch (lang) {
//     case 'shellscript':
//       return 'shell'
//     case 'javascript-typescript':
//       return 'tsx'
//     default:
//       return lang
//   }
// }
