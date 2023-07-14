# Code Completion

## Basic Usage

Tabby will show inline suggestions when you stop typing. You can press the `Tab` key to accept the current suggestion.

![Demo](https://tabbyml.github.io/tabby/img/demo.gif)

## More Actions

Hover over the inline suggestion text to see the toolbar providing more actions.

![InlineCompletionToolbar](./toolbar.png)

### Cycling Through Choices

When multiple choices are available, you can cycle through them by pressing `Alt + [` and `Alt + ]`.

### Partially Accept by Word or by Line

If you want to partially accept a code suggestion, you can accept it by word or by line. Press `Ctrl + RightArrow` to accept next word. Click `...` button on the toolbar, and you will find the option to accept the suggestion by line.

## Keybindings

You can select a keybinding profile in the [settings](command:tabby.openSettings), or customize your own [keybindings](command:tabby.openKeybindings).

|                | Accept Full Completion | Accept Next Word  | Accept Next Line |
| -------------: | :--------------------: | :---------------: | :--------------: |
|  _tabby-style_ |       Ctrl + Tab       | Ctrl + RightArrow |       Tab        |
| _vscode-style_ |          Tab           | Ctrl + RightArrow |        -         |
