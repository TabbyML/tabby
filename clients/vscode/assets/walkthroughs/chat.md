# Chat

**Note**: Chat features are only available when your server supports them. You need to launch the server with the chat model option enabled, for example, `--chat-model Mistral-7B`.

## Chat View

You can start a session in the chat view from the activity bar.

![Chat View](./chatView.png)

### Specific Questions about Selected Code

Select some code, open the command palette and search for command such as [Tabby: Explain This](command:tabby.chat.explainCodeBlock) to ask a question about the selected code.

![Chat View With Selected](./chatViewWithSelected.png)

## Use the Chat Feature to Edit the Code Directly

If you want to use the chat feature to edit the code directly, you can use the [Tabby: Edit...](command:tabby.chat.edit.start) command (shortcut: `Ctrl/Cmd+I`). You can input your request or select a preset command, and Tabby will start editing to meet your needs.

![Chat Edit](./chatEdit.png)

Once the editing is completed, you can select `Accept`(shortcut: `Ctrl/Cmd+Enter`) or `Discard` (shortcut: `Esc`) to deal with the changes You can also use `Esc` to stop the ongoing editing.
Note that the shortcuts require your cursor to be positioned on the header line of the editing block, and the cursor is positioned there by default when the editing is started.

![Chat Edit Completed](./chatEditCompleted.png)
